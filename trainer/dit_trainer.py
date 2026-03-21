import torch
import torch.nn.functional as F
from tqdm import tqdm
from random import random
import wandb
import os

from model.diffusion import Diffusion
from model.gmm import get_mixture_coef, sample_from_params, get_loss
from utils.visual import plot_line, plot_line_cv2, plot_line_cv2_new
from utils.mask import build_prefix_mask_from_char_points
from utils.ddp import reduce_loss, is_main_process, ddp_print, ddp_log_to_wandb

def train_dit_one_epoch(dit, vae, train_loader, optimizer, scheduler, epoch, num_epochs, device, cls_free=False):
    dit.train()
    vae.eval()
    vae = vae.module if hasattr(vae, 'module') else vae
    
    diffusion = Diffusion(
        noise_steps=1000,
        schedule_type='cosine',
        device=device
    )

    max_norm = 1.0

    latent_drop_prob = 0.2
    cond_drop_prob = 0.1

    steps_per_epoch = len(train_loader)
    for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}", unit="batch"):
        data, mask, text_idx, char_points_idx, writer_id = batch
        data, mask, text_idx, writer_id = data.to(device), mask.to(device), text_idx.to(device), writer_id.to(device)

        batch_size = data.size(0)
        global_step = epoch * steps_per_epoch + i + 1

        data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

        with torch.no_grad():
            feat = vae.encode(data)[0]

        latent_mask, latent_padding_mask, prefix_label_mask = build_prefix_mask_from_char_points(
            char_points_idx=char_points_idx,
            mask=mask,
            compression_factor=8,
            prefix_ratio=0.3
        )

        final_noise_mask = latent_mask * latent_padding_mask  # [B, T_latent]
        
        t = diffusion.sample_timesteps(batch_size, finetune=False).to(device)

        x_t_cond, x_t, _ = diffusion.noise_images(feat, t, latent_mask)

        if cls_free:
            drop_cond = random() < latent_drop_prob  # drop prefix latent feat
            if random() < cond_drop_prob:
                drop_cond = True
                drop_text = True
            else:
                drop_text = False
        else:
            drop_text=False
            drop_cond=False

        x_pred = dit(x=x_t_cond, noise=x_t, text=text_idx, time=t, mask=latent_padding_mask, drop_text=drop_text, drop_cond=drop_cond)
        
        # mse loss
        loss = F.mse_loss(x_pred, feat.permute(0, 2, 1), reduction="none")
        mse_loss = loss[final_noise_mask.bool()].mean()

        with torch.no_grad():
            x_mix = x_pred.permute(0, 2, 1) * (final_noise_mask.unsqueeze(1)) + (feat.detach()) * (1 - final_noise_mask).unsqueeze(1)
            ctc_loss = vae.get_ocr_loss(x_mix, text_idx, latent_padding_mask)
            style_loss = vae.get_style_loss(x_mix, writer_id, latent_padding_mask)

        total_loss = mse_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm)
        optimizer.step()

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # DDP
        total_loss = reduce_loss(total_loss)
        mse_loss = reduce_loss(mse_loss)
        ctc_loss = reduce_loss(ctc_loss)
        style_loss = reduce_loss(style_loss)

        if is_main_process():
            ddp_print(f"Epoch [{epoch + 1}/{num_epochs}], Current_lr: {current_lr:.6f}, "
                    f"Step [{i + 1}/{steps_per_epoch}], Total_loss:{total_loss.item():.6f}, "
                    f"MSE:{mse_loss.item():.6f}, CTC_loss:{ctc_loss.item():.6f}, Style_loss:{style_loss.item():.6f}")

            ddp_log_to_wandb({
                "learning_rate": current_lr,
                "Total_loss": total_loss.item(),
                "MSE_loss": mse_loss.item(),
                "CTC_loss": ctc_loss.item(),
                "Style_loss": style_loss.item()
            }, step=global_step)


def tune_dit_one_epoch(dit, vae, train_loader, optimizer, scheduler, epoch, num_epochs, device):
    dit.train()
    vae.eval()
    vae = vae.module if hasattr(vae, 'module') else vae
    
    diffusion = Diffusion(
        noise_steps=1000,
        schedule_type='cosine',
        device=device
    )

    max_norm = 1.0

    steps_per_epoch = len(train_loader)
    for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}", unit="batch"):
        data, mask, text_idx, char_points_idx, writer_id = batch
        data, mask, text_idx, writer_id = data.to(device), mask.to(device), text_idx.to(device), writer_id.to(device)

        batch_size = data.size(0)
        global_step = epoch * steps_per_epoch + i + 1

        data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

        feat = vae.encode(data)[0].permute(0, 2, 1)

        latent_mask, latent_padding_mask, prefix_mask = build_prefix_mask_from_char_points(
            char_points_idx=char_points_idx,
            mask=mask,
            compression_factor=8,
            prefix_ratio=0.3
        )

        final_noise_mask = latent_mask * latent_padding_mask  # [B, T_latent]
        
        t = diffusion.sample_timesteps(batch_size, finetune=True).to(device)

        x_t, noise = diffusion.noise_images(feat, t)
        
        x_0_final, x_0, noise_0 = diffusion.train_ddim(dit, batch_size, x_t, feat, text_idx, latent_mask, latent_padding_mask, t, sampling_timesteps=5, eta=0.0, drop_text=False, drop_cond=False)

        # mse loss
        loss = F.mse_loss(x_0_final, feat, reduction="none")
        mse_loss = loss[final_noise_mask.bool()].mean()

        loss_noise = F.mse_loss(noise_0, noise, reduction="none")
        mse_loss_noise = loss_noise[final_noise_mask.bool()].mean()

        total_loss = mse_loss + mse_loss_noise
        
        # update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm)
        optimizer.step()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        total_loss = reduce_loss(total_loss)
        mse_loss = reduce_loss(mse_loss)
        mse_loss_noise = reduce_loss(mse_loss_noise)

        if is_main_process():
            ddp_print(f"Epoch [{epoch + 1}/{num_epochs}], Current_lr: {current_lr:.6f}, Step [{i + 1}/{steps_per_epoch}], Total_loss:{total_loss:.6f}, MSE:{mse_loss:.6f}, Noise_MSE:{mse_loss_noise:.6f}")

            ddp_log_to_wandb({
                "learning_rate": current_lr,
                "Total_loss": total_loss.item(),
                "MSE_loss": mse_loss.item(),
                "Noise_MSE": mse_loss_noise.item(),
            }, step=global_step)


def val_dit_one_batch(dit, vae, val_loader, device, save_path):
    dit.eval()
    vae.eval()
    vae = vae.module if hasattr(vae, 'module') else vae

    diffusion = Diffusion(
        noise_steps=1000,
        schedule_type='cosine',
        device=device
    )

    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for p, batch in tqdm(enumerate(val_loader), desc="val", unit="batch"):
            data, mask, text_idx, char_points_idx, writer_id = batch
            data, mask, text_idx = data.to(device), mask.to(device), text_idx.to(device)

            batch_size = data.size(0)

            data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

            feat = vae.encode(data)[0].permute(0, 2, 1)

            latent_mask, latent_padding_mask, prefix_label_mask = build_prefix_mask_from_char_points(
                char_points_idx=char_points_idx,
                mask=mask,  # e.g., mask from dataloader
                compression_factor=8,
                prefix_ratio=0.3
            )
            final_noise_mask = latent_mask * latent_padding_mask  # [B, T_latent]

            x_pred = diffusion.ddim_sample(dit, batch_size, feat, text_idx, latent_mask, latent_padding_mask, sampling_timesteps=5, eta=0.0)

            loss = F.mse_loss(x_pred, feat, reduction="none")
            mse_loss = loss[final_noise_mask.bool()].mean()

            x_mix = x_pred.permute(0, 2, 1) * (final_noise_mask.unsqueeze(1)) + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)

            # decoder and visualized
            output = vae.decode(x_mix)

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

                recon = sample_from_params(params, temp=0.1, max_seq_len=seq_len, greedy=True) # T 5
                plot_line(data[i], save_path=f"{save_path}/gt_{p}_{i}.png", title='gt')
                plot_line(recon, save_path=f"{save_path}/recon_{p}_{i}.png", title='recon')
            
            break

def infer_diffink(dit, vae, val_loader, device, save_path):
    dit.eval()
    vae.eval()
    vae = vae.module if hasattr(vae, 'module') else vae

    diffusion = Diffusion(
        noise_steps=1000,
        schedule_type='cosine',
        device=device
    )

    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for p, batch in tqdm(enumerate(val_loader), desc="val", unit="batch"):
            data, mask, text_idx, char_points_idx, writer_id = batch
            data, mask, text_idx = data.to(device), mask.to(device), text_idx.to(device)

            batch_size = data.size(0)

            data = data.permute(0, 2, 1)    # [B, T, 5] → [B, 5, T]

            feat = vae.encode(data)[0].permute(0, 2, 1)

            latent_mask, latent_padding_mask, prefix_label_mask = build_prefix_mask_from_char_points(
                char_points_idx=char_points_idx,
                mask=mask,  # e.g., mask from dataloader
                compression_factor=8,
                prefix_ratio=0.3
            )
            final_noise_mask = latent_mask * latent_padding_mask  # [B, T_latent]

            x_pred = diffusion.ddim_sample(dit, batch_size, feat, text_idx, latent_mask, latent_padding_mask, sampling_timesteps=20, eta=0.0, cfg_scale=1.0)

            x_mix = x_pred.permute(0, 2, 1) * (final_noise_mask.unsqueeze(1)) + feat.permute(0, 2, 1) * (1 - final_noise_mask).unsqueeze(1)

            # decoder and visualized
            output = vae.decode(x_mix)

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
                plot_line_cv2_new(data[i], save_path=f"{save_path}/gt_{p * batch_size + i}.png", canvas_height=256, padding=20, line_thickness=2, max_dist=200)
                plot_line_cv2_new(recon, save_path=f"{save_path}/recon_{p * batch_size + i}.png", canvas_height=256, padding=20, line_thickness=2, max_dist=200)
            
            # break

