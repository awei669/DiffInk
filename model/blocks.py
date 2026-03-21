import torch
import torch.nn as nn
import torch.nn.functional as F

# for vae encoder and decoder

class Residual(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            Residual(in_channels, hidden_channels) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=5, hidden_dims=[128, 256, 384]):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1)
        self.res_stack_1 = ResidualStack(hidden_dims[0], hidden_dims[0] // 2, num_layers=4)

        self.conv_2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3, stride=2, padding=1)
        self.res_stack_2 = ResidualStack(hidden_dims[1], hidden_dims[1] // 2, num_layers=4)

        self.conv_3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, stride=2, padding=1)
        self.res_stack_3 = ResidualStack(hidden_dims[2], hidden_dims[2] // 2, num_layers=4)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_stack_1(x)
        x = self.conv_2(x)
        x = self.res_stack_2(x)
        x = self.conv_3(x)
        x = self.res_stack_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dims=[384, 256, 128]):
        super().__init__()
        self.res_stack_1 = ResidualStack(hidden_dims[0], hidden_dims[0] // 2, num_layers=4)
        self.deconv_1 = nn.ConvTranspose1d(hidden_dims[0], hidden_dims[1], kernel_size=4, stride=2, padding=1)

        self.res_stack_2 = ResidualStack(hidden_dims[1], hidden_dims[1] // 2, num_layers=4)
        self.deconv_2 = nn.ConvTranspose1d(hidden_dims[1], hidden_dims[2], kernel_size=4, stride=2, padding=1)

        self.res_stack_3 = ResidualStack(hidden_dims[2], hidden_dims[2] // 2, num_layers=4)
        self.deconv_3 = nn.ConvTranspose1d(hidden_dims[2], 128, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res_stack_1(x)
        x = self.deconv_1(x)
        x = self.res_stack_2(x)
        x = self.deconv_2(x)
        x = self.res_stack_3(x)
        x = self.deconv_3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=123, num_layers=4, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # self.input_norm = nn.LayerNorm(hidden_dim)
        # self.output_norm = nn.LayerNorm(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, padding_mask=None):
        x = x.permute(0, 2, 1)  # [B, C, T] → [B, T, C]
        x = self.input_proj(x)
        # x = self.input_norm(x)  # <--- new
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # x = self.output_norm(x)  # <--- new
        x = self.fc(x)
        return x.permute(0, 2, 1)  # [B, T, D] → [B, D, T]
