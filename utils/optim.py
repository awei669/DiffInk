import torch.optim as optim

import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_optimizer_and_scheduler(model, config):
    base_lr = config.get("base_lr", 5e-4)
    betas = config.get("betas", (0.9, 0.99))
    weight_decay = config.get("weight_decay", 1e-4)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=betas, weight_decay=weight_decay)

    T_max = config.get("num_epochs", 100)
    eta_min = config.get("min_lr", 1e-5)

    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    return optimizer, scheduler

def dit_build_optimizer_and_scheduler(model, config, num_step_per_epoch):
    base_lr = config.get("base_lr", 1e-4)
    min_lr = config.get("min_lr", 1e-6)
    betas = config.get("betas", (0.9, 0.99))
    weight_decay = config.get("weight_decay", 1e-4)
    num_epochs = config.get("num_epochs", 1000)

    total_steps = num_epochs * num_step_per_epoch
    warmup_steps = config.get("warmup_steps", total_steps* 0.05)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=betas, weight_decay=weight_decay)

    # ---- 自定义学习率策略 ----
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1. + math.cos(math.pi * progress))
            return min_lr / base_lr + (1. - min_lr / base_lr) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler