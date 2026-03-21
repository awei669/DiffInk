import torch
import math
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

class Diffusion:
    def __init__(self, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None, schedule_type='linear'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_offset = noise_offset
        self.device = device
        self.schedule_type = schedule_type.lower()

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        if self.schedule_type == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.schedule_type == 'cosine':
            return self.cosine_beta_schedule(self.noise_steps)
        else:
            raise ValueError(f"Unsupported schedule_type: {self.schedule_type}")

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def predict_start_from_noise(self, x_t, t, noise_pred):
        if isinstance(t, int):
            alpha_hat = self.alpha_hat[t]
            alpha_hat = torch.tensor(alpha_hat, device=x_t.device)
        else:
            alpha_hat = self.alpha_hat[t].to(x_t.device)

        if alpha_hat.dim() == 1:
            alpha_hat = alpha_hat[:, None, None]

        sqrt_alpha_hat = torch.sqrt(alpha_hat)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)

        x0_pred = (x_t - sqrt_one_minus_alpha_hat * noise_pred) / sqrt_alpha_hat

        return x0_pred

    def noise_images(self, x=None, t=None, noise_mask=None):
        if x.shape[1] != 384: x = x.permute(0, 2, 1)
        B, C, T = x.shape
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]  # [B, 1, 1]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]

        noise = torch.randn_like(x)

        if self.noise_offset != 0:
            noise += self.noise_offset * torch.randn(B, C, 1).to(x.device)

        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise

        # 默认全加噪
        if noise_mask is None:
            return x_t.permute(0, 2, 1), noise.permute(0, 2, 1)
        else:
            if noise_mask.dim() == 2:
                noise_mask = noise_mask.unsqueeze(1).to(dtype=torch.float32)  # [B, 1, T]
            x_t_cond = x * (1 - noise_mask) + x_t * noise_mask  # 混合结果
            return x_t_cond.permute(0, 2, 1), x_t.permute(0, 2, 1), noise.permute(0, 2, 1)

    def sample_timesteps(self, n, finetune=False):
        if finetune:
            return torch.randint(6, self.noise_steps, (n,))
        else:
            return torch.randint(0, self.noise_steps, (n,))
    
    def train_ddim(self, dit_model, n, x, cond, text, cond_mask, padding_mask, total_t, sampling_timesteps=5, eta=0.0, drop_text=False, drop_cond=False):
        if cond_mask.dim() == 2:
            cond_mask = cond_mask.unsqueeze(-1).to(dtype=torch.float32)  # [B, T, 1]

        dit_model.train()

        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = [-1] + [i / sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x_start = None
        x_start_list = []
        eps_list = []
        for time, time_next in time_pairs:
            time = (total_timesteps * time).long().to(self.device)
            time_next = (total_timesteps * time_next).long().to(self.device)

            x_cond = cond * (1 - cond_mask) + x * cond_mask  # 条件注入

            # 包装 forward 用于 checkpoint
            def forward_fn(x_in, noise, text_in, time_in, mask_in, drop_text, drop_cond):
                return dit_model(x=x_in, noise=noise, text=text_in, time=time_in, mask=mask_in, drop_text=drop_text, drop_cond=drop_cond)

            # 使用 checkpoint 调用模型
            x_start = checkpoint(forward_fn, x_cond, x, text, time, padding_mask, drop_text, drop_cond)
            x_start_list.append(x_start)

            # 获取当前与下一步的 alpha_hat
            alpha_hat_t = self.alpha_hat[time][:, None, None]          # [B, 1, 1]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None]  # [B, 1, 1]
            beta_t = self.beta[time][:, None, None]

            # 推出 epsilon（因为预测的是 x0）
            eps = (x - alpha_hat_t.sqrt() * x_start) / (1 - alpha_hat_t).sqrt()
            eps_list.append(eps)

            # 最后一步：直接输出预测结果
            if time_next[0] < 0:
                x = x_start
                continue

            # 计算 sigma 和系数
            sigma = eta * ((1 - alpha_hat_t / alpha_hat_next) * beta_t).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()

            # 可控噪声（eta 控制采样随机性）
            if eta > 0:
                sample_noise = torch.randn_like(x)
            else:
                sample_noise = torch.zeros_like(x)

            # 更新 xt-1
            x = alpha_hat_next.sqrt() * x_start + c * eps + sigma * sample_noise

        return x, x_start_list[0], eps_list[0]


    @torch.no_grad()
    def ddim_sample(self, dit_model, n, cond, text, cond_mask, padding_mask, sampling_timesteps=5, eta=0.0, cfg_scale=1.0):
        if cond_mask.dim() == 2:
            cond_mask = cond_mask.unsqueeze(-1)
        cond_mask = cond_mask.to(dtype=torch.float32, device=self.device)

        dit_model.eval()

        x = torch.randn_like(cond)

        total_timesteps = self.noise_steps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            time_tensor = torch.full((n,), time, device=self.device, dtype=torch.long)

            # 当前 step 的两种版本
            x_uncond = x
            x_cond = cond * (1 - cond_mask) + x * cond_mask

            # cond branch
            x_start_cond = dit_model(
                x=x_cond,
                noise=x_uncond,
                text=text,
                time=time_tensor,
                mask=padding_mask,
                drop_text=False,
                drop_cond=False
            )

            # uncond branch
            x_start_uncond = dit_model(
                x=x_cond,
                noise=x_uncond,
                text=text,
                time=time_tensor,
                mask=padding_mask,
                drop_text=False,
                drop_cond=True
            )

            x_start = x_start_uncond + cfg_scale * (x_start_cond - x_start_uncond)

            alpha_hat_t = self.alpha_hat[time_tensor][:, None, None]

            if time_next < 0:
                x = x_start
                continue

            time_next_tensor = torch.full((n,), time_next, device=self.device, dtype=torch.long)
            alpha_hat_next = self.alpha_hat[time_next_tensor][:, None, None]

            eps = (x - alpha_hat_t.sqrt() * x_start) / (1 - alpha_hat_t).clamp_min(1e-8).sqrt()

            sigma = eta * torch.sqrt(
                (
                    (1 - alpha_hat_next) / (1 - alpha_hat_t).clamp_min(1e-8)
                    * (1 - alpha_hat_t / alpha_hat_next)
                ).clamp_min(0.0)
            )
            c = torch.sqrt((1 - alpha_hat_next - sigma ** 2).clamp_min(0.0))

            if eta > 0:
                sample_noise = torch.randn_like(x)
            else:
                sample_noise = torch.zeros_like(x)

            x = alpha_hat_next.sqrt() * x_start + c * eps + sigma * sample_noise

        return x
        
