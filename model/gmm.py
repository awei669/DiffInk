import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

INF_MIN = 1e-8

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, alpha=None):
        """
        :param inputs: 形状为 (batch_size, seq_len, num_classes) 的预测值
        :param targets: 形状为 (batch_size, seq_len) 的目标标签
        :param alpha: 形状为 (batch_size, num_classes) 的类别权重张量
        :return: 计算后的焦损
        """
        batch_size, seq_len, num_classes = inputs.size()

        inputs_reshaped = inputs.reshape(-1, num_classes)  # (batch_size * seq_len, num_classes)
        targets_reshaped = targets.reshape(-1)  # (batch_size * seq_len)

        ce_loss_per_element = F.cross_entropy(inputs_reshaped, targets_reshaped, reduction='none')

        pt = torch.exp(-ce_loss_per_element)

        if alpha is not None:
            alpha = alpha.to(inputs.device)
            target_class_weights = alpha[torch.arange(batch_size).unsqueeze(-1), targets]
            focal_loss_per_element = target_class_weights.reshape(-1) * (1 - pt) ** self.gamma * ce_loss_per_element
        else:
            focal_loss_per_element = (1 - pt) ** self.gamma * ce_loss_per_element

        focal_loss_per_sample = focal_loss_per_element.reshape(batch_size, seq_len)

        if self.reduction == 'mean':
            return focal_loss_per_sample.mean()
        elif self.reduction == 'sum':
            return focal_loss_per_sample.sum()
        else:
            return focal_loss_per_sample


def compute_sample_class_weights(targets, num_classes):
    B, T = targets.size()
    counts = torch.zeros((B, num_classes), device=targets.device)
    for i in range(B):
        counts[i] = torch.bincount(targets[i], minlength=num_classes)
    total = counts.sum(dim=1, keepdim=True)
    total[total == 0] = 1
    weights = total / (counts + INF_MIN)
    weights[counts == 0] = 0
    return weights


def torch_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    norm1 = x1 - mu1
    norm2 = x2 - mu2
    s1s2 = s1 * s2
    z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / s1s2
    neg_rho = torch.clamp(1 - rho ** 2, min=INF_MIN)
    result = torch.exp(-z / (2 * neg_rho))
    denom = 2 * np.pi * s1s2 * torch.sqrt(neg_rho)
    return result / denom


def get_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits,
             x1_data, x2_data, pen_data, focal_loss_reduction='mean'):
    criterion_ce = FocalLoss(gamma=2.0, reduction=focal_loss_reduction)

    # GMM Loss
    gmm_pdf = torch_2d_normal(x1_data, x2_data, mu1, mu2, sigma1, sigma2, corr)
    weighted_pdf = pi * gmm_pdf
    log_prob = -torch.log(torch.sum(weighted_pdf, dim=1, keepdim=True) + INF_MIN)  # [B, 1, T]
    gmm_loss = log_prob.squeeze(1)  # [B, T]

    # Pen state CE loss
    pen_logits = pen_logits.permute(0, 2, 1)  # [B, T, 3]
    pen_targets = torch.argmax(pen_data, dim=1)  # [B, T]
    class_weights = compute_sample_class_weights(pen_targets, num_classes=3)
    ce_loss = criterion_ce(pen_logits, pen_targets, alpha=class_weights)

    return gmm_loss, ce_loss


def get_mixture_coef(output, num_mixture):
    pen_logits = output[:, :3, :]
    gmm_params = output[:, 3:, :]
    pi, mu1, mu2, sigma1, sigma2, corr = torch.split(gmm_params, num_mixture, dim=1)

    pi = torch.softmax(pi, dim=1)
    pen = torch.softmax(pen_logits, dim=1)
    sigma1 = torch.exp(sigma1)
    sigma2 = torch.exp(sigma2)
    corr = torch.tanh(corr)

    return [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]

def get_mixture_coef_max(output, num_mixture):
    pen_logits = output[:, :3, :]             # [B, 3, T]
    gmm_params = output[:, 3:, :]             # [B, num_mixture * 6, T]

    pi, mu1, mu2, sigma1, sigma2, corr = torch.split(gmm_params, num_mixture, dim=1)

    pi = torch.softmax(pi, dim=1)
    sigma1 = F.softplus(sigma1) + 1e-3        # avoid near-zero
    sigma2 = F.softplus(sigma2) + 1e-3
    corr = torch.tanh(corr)

    pen = torch.softmax(pen_logits, dim=1)    # optional: if used for sampling

    return [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]



def sample_gaussian_2d(mu1, mu2, s1, s2, rho, sqrt_temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    s1 *= sqrt_temp ** 2
    s2 *= sqrt_temp ** 2
    cov = [[s1 * s1, rho * s1 * s2],
           [rho * s1 * s2, s2 * s2]]
    return np.random.multivariate_normal([mu1, mu2], cov)


def sample_from_params(params, temp=0.1, max_seq_len=400, greedy=False):
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = params
    num_mixture, seq_len = o_pi.shape
    strokes = np.zeros((seq_len, 5), dtype=np.float32)

    for step in range(max_seq_len):
        eos = [0, 0, 0]
        idx = torch.distributions.Categorical(o_pi[:, step]).sample().item()
        x1, x2 = sample_gaussian_2d(
            o_mu1[idx, step].item(),
            o_mu2[idx, step].item(),
            o_sigma1[idx, step].item(),
            o_sigma2[idx, step].item(),
            o_corr[idx, step].item(),
            sqrt_temp=np.sqrt(temp),
            greedy=greedy
        )
        eos[np.argmax(o_pen[:, step].cpu().numpy())] = 1
        strokes[step] = [x1, x2] + eos

    return strokes
