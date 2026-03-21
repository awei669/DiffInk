import sys
sys.path.append('/home/pw/hdd_3/Project/TTH_new')

import os
import torch
import wandb
from datetime import datetime
from torch.nn.parallel import DataParallel

from model.vae import VAE
from dataset import build_datasets_and_loaders_ddp
from utils.optim import dit_build_optimizer_and_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import ModelConfig, save_checkpoint, load_config_from_yaml, set_seed
from trainer.vae_trainer import train_vae_one_epoch, val_vae_one_batch

from model.dit import TextEmbedding, InputEmbedding, DiT

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def train_vae(task_name, time):
    # === Setup DDP ===
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    config_path = "./configs/vae_config.yaml"
    config_dict = load_config_from_yaml(config_path)

    train_loader, val_loader, config_dict = build_datasets_and_loaders_ddp(config_dict, ddp=True)
    config = ModelConfig(config_dict)

    # === init ===
    set_seed(42)
    vae = VAE(config).to(device)

    optimizer, scheduler = dit_build_optimizer_and_scheduler(vae, config_dict, len(train_loader))

    # === load vae ====
    vae_model_path = config_dict.get("vae_model_path")
    if vae_model_path and os.path.exists(vae_model_path):
        print(f"Resuming VAE from checkpoint: {vae_model_path}")

        ckpt = torch.load(vae_model_path, map_location=device)
        state_dict = ckpt["model_state_dict"]

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = strip_module_prefix(state_dict)

        model_state_dict = vae.state_dict()
        skip_prefixes =[]

        filtered_dict = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in skip_prefixes)
            and k in model_state_dict and v.shape == model_state_dict[k].shape
        }

        skipped_keys = [k for k in state_dict if k not in filtered_dict]
        print(f"Skipped loading weights for: {skipped_keys}")

        vae.load_state_dict(filtered_dict, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
    else:
        start_epoch = 0

    vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=True)

    if local_rank == 0:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(
            project="train_vae_latest",     
            name=task_name,                 
            config={                        
                "config": config_dict,
                "vae_model": vae,
            }
        )

    num_epochs = config_dict.get("num_epochs", 100)
    save_every = config_dict.get("save_every", 5)
    val_every = config_dict.get("val_every", 5)
    output_path = config_dict.get("output_base", './')
    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch)
        train_vae_one_epoch(vae, config, train_loader, optimizer, scheduler, epoch, num_epochs, device, ddp=True)

        if local_rank == 0:
            if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                save_checkpoint(vae, optimizer, scheduler, epoch, 'vae', f'{output_path}/{task_name}_{time}/ckpt')
            if (epoch + 1) % val_every == 0 or (epoch + 1) == num_epochs:
                val_vae_one_batch(vae, val_loader, epoch, device, f'{output_path}/{task_name}_{time}/imgs/epoch_{epoch + 1}')

    dist.destroy_process_group()

if __name__ == "__main__":
    now = datetime.now()
    print("Time:", now)
    formatted_time = now.strftime("%m_%d_%H_%M")
    task_name = 'ink_vae'
    train_vae(task_name, formatted_time)