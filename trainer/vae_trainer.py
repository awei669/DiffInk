
import os
import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from model.gmm import get_mixture_coef, get_mixture_coef_max, get_loss, sample_from_params
from utils.visual import plot_line_cv2
from utils.mask import downsample_mask
from utils.ddp import reduce_loss, is_main_process, ddp_print, ddp_log_to_wandb

def train_vae_one_epoch(vae, config, train_loader, optimizer, scheduler, epoch, num_epochs, device, ddp=False):
    vae.train()
    total_gmm_loss = 0.0
    total_ce_loss = 0.0
    total_kl_loss = 0.0
    total_ctc_loss = 0.0
    total_style_loss = 0.0
    total_total_loss = 0.0

    gmm_weight, kl_weight, pen_weight, ctc_weight, style_weight = config.gmm_weight, config.kl_weight, config.pen_weight, config.ctc_weight, config.style_weight
    get_ctc_loss = ctc_weight != 0
    get_style_loss = style_weight != 0

    steps_per_epoch = len(train_loader)
    for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}", unit="batch"):
        global_step = epoch * steps_per_epoch + i + 1
        data, mask, text_idx, char_points_idx, writer_id = batch
        data, mask, text_idx, writer_id = data.to(device), mask.to(device), text_idx.to(device), writer_id.to(device)

        optimizer.zero_grad(set_to_none=True)

        pad_mask = downsample_mask(mask, compression_factor=8)

        data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

        output, ctc_loss, kl_loss, style_loss = vae(data, pad_mask, text_idx, writer_id, get_ctc_loss, get_style_loss)

        # 获取连续值和 one-hot 向量的真值
        x_true = data[:, :1, :]
        y_true = data[:, 1:2, :]
        pen_state_true = data[:, 2:, :]
        
        # 获取 GMM 参数
        pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = get_mixture_coef_max(output, num_mixture=20)

        # 计算损失
        loss_d, loss_c = get_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits, x_true, y_true, pen_state_true)

        gmm_loss = (loss_d * mask).sum() / mask.sum()
        ce_loss = loss_c
        kl_loss = kl_loss.mean()
        ctc_loss = ctc_loss.mean()
        style_loss = style_loss.mean()

        total_loss = gmm_loss * gmm_weight + ce_loss * pen_weight + kl_loss * kl_weight

        if ctc_weight != 0:
            total_loss += ctc_loss * ctc_weight
        if style_weight != 0:
            total_loss += style_loss * style_weight

        if ddp:
            is_finite = torch.isfinite(total_loss).to(dtype=torch.float32)
            torch.distributed.all_reduce(is_finite, op=torch.distributed.ReduceOp.MIN)
            skip_step = (is_finite.item() == 0)
        else:
            skip_step = (not torch.isfinite(total_loss).item())

        if skip_step:
            if (not ddp) or torch.distributed.get_rank() == 0:
                def safe_item(x):
                    try:
                        return x.item()
                    except Exception:
                        return float("nan")

                print(
                    f"[Warning] Skipping step at epoch {epoch}, step {i}: "
                    f"Total Loss: {safe_item(total_loss):.4f}, "
                    f"GMM Loss: {safe_item(gmm_loss):.4f}, "
                    f"CE Loss: {safe_item(ce_loss):.4f}, "
                    f"KL Loss: {safe_item(kl_loss):.4f}, "
                    f"CTC Loss: {safe_item(ctc_loss):.4f}, "
                    f"Style Loss: {safe_item(style_loss):.4f}"
                )

            # 释放所有仍可能持有计算图引用的 tensor
            del total_loss
            del gmm_loss, ce_loss, kl_loss, ctc_loss, style_loss
            del loss_d, loss_c
            del pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits
            del x_true, y_true, pen_state_true
            del output
            del pad_mask
            del data, mask, text_idx, writer_id, char_points_idx, batch

            torch.cuda.empty_cache()   # 仅异常分支用一次即可
            continue

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
            
        if ddp:
            total_loss = reduce_loss(total_loss)
            gmm_loss = reduce_loss(gmm_loss)
            ce_loss = reduce_loss(ce_loss)
            kl_loss = reduce_loss(kl_loss)

            if get_ctc_loss:
                ctc_loss = reduce_loss(ctc_loss)
            else:
                ctc_loss = torch.tensor(0.0, device=total_loss.device)

            if get_style_loss:
                style_loss = reduce_loss(style_loss)
            else:
                style_loss = torch.tensor(0.0, device=total_loss.device)

            # 每 50 step 输出一次
            if is_main_process() and global_step % 50 == 0:
                ddp_print(
                    f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Step [{i}/{steps_per_epoch}], '
                    f'LR: {current_lr:.6f}, '
                    f'Total Loss: {total_loss.item():.4f}, '
                    f'GMM Loss: {gmm_loss.item():.4f}, '
                    f'CE Loss: {ce_loss.item():.4f}, '
                    f'KL Loss: {kl_loss.item():.4f}, '
                    f'CTC Loss: {ctc_loss.item():.4f}, '
                    f'Style Loss: {style_loss.item():.4f}'
                )
        
        else:
            print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr}, Total Loss: {total_loss.item():.4f}, GMM Loss: {gmm_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, CTC Loss: {ctc_loss.item():.4f}, Style Loss: {style_loss.item():.4f}')
        
        # 正常 step 结束后统一清理
        del total_loss
        del gmm_loss, ce_loss, kl_loss, ctc_loss, style_loss
        del loss_d, loss_c
        del pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits
        del x_true, y_true, pen_state_true
        del output
        del pad_mask
        del data, mask, text_idx, writer_id, char_points_idx, batch

def val_vae_one_batch(model, val_loader, epoch, device, save_path):
    model.eval()
    model = model.module if hasattr(model, 'module') else model
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for p, batch in tqdm(enumerate(val_loader), desc=f"Epoch {epoch + 1}", unit="batch"):
            data, mask, labels, char_points_idx, writer_id = batch
            data, mask = data.to(device), mask.to(device)

            data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

            output = model.val(data)

            # sample from gmm
            pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits = get_mixture_coef(output, num_mixture=20)

            for i in range(data.size(0)):
                sample_idx = i
                seq_len = int(mask[i].sum().item())
                params = [
                    pi[sample_idx].cpu(),  # 混合系数
                    mu1[sample_idx].cpu(),  # 第一个分量的均值
                    mu2[sample_idx].cpu(),  # 第二个分量的均值
                    sigma1[sample_idx].cpu(),  # 第一个分量的标准差
                    sigma2[sample_idx].cpu(),  # 第二个分量的标准差
                    corr[sample_idx].cpu(),  # 相关系数 
                    pen[sample_idx].cpu()  # 结束状态的概率
                ]

                # 生成样本
                recon = sample_from_params(params, temp=0.1, max_seq_len=seq_len, greedy=True) # T 5
                plot_line_cv2(data[i], save_path=f"{save_path}/gt_{p}_{i}.png", canvas_height=256, padding=20, line_thickness=2)
                plot_line_cv2(recon, save_path=f"{save_path}/recon_{p}_{i}.png", canvas_height=256, padding=20, line_thickness=2)

            break

