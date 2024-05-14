import torch
import torch.nn as nn

from src.modeling.base import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.gelu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

class ResAutoEncoder(BaseModel):
    def __init__(self, model_name='residual_ae'):
        super(ResAutoEncoder, self).__init__(model_name)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            ResidualBlock(128),
            nn.Conv2d(128, 32, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            ResidualBlock(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            ResidualBlock(16),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.GELU(),
            ResidualBlock(32),
            nn.ConvTranspose2d(
                32, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.GELU(),
            ResidualBlock(128),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x