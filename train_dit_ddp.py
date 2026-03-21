import sys
sys.path.append('/home/pw/hdd_3/Project/TTH_new')

import os
import torch
import wandb
from copy import deepcopy
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model.vae import VAE
from dataset import build_datasets_and_loaders_ddp
from utils.optim import dit_build_optimizer_and_scheduler
from utils.utils import ModelConfig, save_checkpoint, load_config_from_yaml, set_seed
from trainer.dit_trainer import train_dit_one_epoch, val_dit_one_batch

from model.dit import TextEmbedding, InputEmbedding, DiT

def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def train_dit_ddp(task_name, time):
    # === Setup DDP ===
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # === 加载配置 ===
    config_path = "./configs/dit_config.yaml"
    config_dict = load_config_from_yaml(config_path)

    # === 构建数据集与加载器（使用 DistributedSampler） ===
    train_loader, val_loader, config_dict = build_datasets_and_loaders_ddp(config_dict, ddp=True)
    config = ModelConfig(config_dict)

    # === 初始化模型 ===
    set_seed(42)
    vae = VAE(config).to(device)
    dit = DiT(config).to(device)

    optimizer, scheduler = dit_build_optimizer_and_scheduler(dit, config_dict, len(train_loader))

    # === 加载 VAE 检查点 ===
    vae_model_path = config_dict.get("vae_model_path")
    if vae_model_path and os.path.exists(vae_model_path):
        print(f"Resuming VAE from checkpoint: {vae_model_path}")
        ckpt = torch.load(vae_model_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = strip_module_prefix(state_dict)
        vae.load_state_dict(state_dict)

    # === 加载 DIT 检查点 ===
    dit_ckpt_path = config_dict.get("dit_resume_ckpt", None)
    if dit_ckpt_path and os.path.exists(dit_ckpt_path):
        print(f"Resuming DIT from checkpoint: {dit_ckpt_path}")
        ckpt = torch.load(dit_ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = strip_module_prefix(state_dict)

        model_state = dit.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        dit.load_state_dict(filtered_state_dict, strict=False)
        start_epoch = 0

        dit.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
    else:
        start_epoch = 0
    
    vae = DDP(vae, device_ids=[local_rank])
    dit = DDP(dit, device_ids=[local_rank])

    # 只在主进程记录 wandb
    if local_rank == 0:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project="train_dit_new",
            name=task_name,
            config={
                "config": config_dict,
                "vae_model": vae,
                "dit_model": dit,
            }
        )

    # === 开始训练 ===
    num_epochs = config_dict.get("num_epochs", 100)
    save_every = config_dict.get("save_every", 5)
    val_every = config_dict.get("val_every", 5)
    output_path = config_dict.get("output_base", './')

    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_dit_one_epoch(dit, vae, train_loader, optimizer, scheduler, epoch, num_epochs, device, True)

        if local_rank == 0:
            if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                save_checkpoint(dit, optimizer, scheduler, epoch, 'dit', f'{output_path}/{task_name}_{time}/ckpt')
            if (epoch + 1) % val_every == 0 or (epoch + 1) == num_epochs:
                val_dit_one_batch(dit, vae, val_loader, device, f'{output_path}/{task_name}_{time}/imgs/epoch_{epoch + 1}')

    dist.destroy_process_group()

if __name__ == "__main__":
    now = datetime.now()
    formatted_time = now.strftime("%m_%d_%H_%M")
    task_name = 'ink_dit'
    train_dit_ddp(task_name, formatted_time)
