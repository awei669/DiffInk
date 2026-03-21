import os
import yaml
import torch
import torch.nn as nn

def load_config_from_yaml(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


class ModelConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)

def save_checkpoint(model, optimizer, scheduler, epoch, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, name + "_epoch_" + str(epoch + 1) + ".pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, ckpt_path)
    print(f"✅ 已保存 checkpoint: {ckpt_path}")

def decode_ctc_output(log_probs, blank=0):
    """
    log_probs: [T, B, C]
    返回：每个样本的预测字符 ID 列表（去掉 blank 和重复）
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1)  # [B, T]
    decoded = []
    for seq in preds:
        prev = -1
        output = []
        for c in seq:
            c = c.item()
            if c != prev and c != blank:
                output.append(c)
            prev = c
        decoded.append(output)
    return decoded


def check_for_nan(model, loss_dict: dict, extra_tensors: dict = None):
    """
    检查训练过程中是否存在 NaN 或 Inf，并打印具体模块或参数名。
    
    Args:
        model: nn.Module 模型实例
        loss_dict: 一个字典，包含所有 loss 分量，例如 {"ctc_loss": ctc, "kl_loss": kl}
        extra_tensors: 你想额外检查的中间变量，例如 {"logits": logits, "mu": mu}
    """

    # 1. 检查各个 loss 分量
    for name, loss in loss_dict.items():
        if not torch.isfinite(loss).all():
            print(f"[⚠️ NaN WARNING] Loss `{name}` contains NaN or Inf → value: {loss.item() if loss.numel() == 1 else 'tensor'}")

    # 2. 检查模型参数和梯度
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"[❌ NaN] Parameter `{name}` contains NaN values!")
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"[❌ NaN] Gradient of `{name}` contains NaN!")
            if torch.isinf(param.grad).any():
                print(f"[❌ Inf] Gradient of `{name}` contains Inf!")

    # 3. 检查额外提供的中间变量
    if extra_tensors:
        for name, tensor in extra_tensors.items():
            if not torch.isfinite(tensor).all():
                print(f"[⚠️ NaN WARNING] Tensor `{name}` contains NaN or Inf.")


def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 cudnn 可复现（关闭 benchmark，开启 deterministic）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[✅] Seed set to {seed}")


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)