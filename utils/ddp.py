import torch
import torch.distributed as dist

def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def reduce_loss(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= dist.get_world_size()
    return tensor

def reduce_loss_dict(loss_dict: dict, average: bool = True) -> dict:
    return {k: reduce_loss(v, average) for k, v in loss_dict.items()}

def ddp_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def ddp_log_to_wandb(metrics: dict, step: int):
    if is_main_process():
        import wandb
        wandb.log(metrics, step=step)
