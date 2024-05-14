import torch
import torch.nn as nn

from src.modeling.base import BaseModel


class LNAutoEncoder(BaseModel):
    def __init__(self, model_name='ln_auto_encoder'):
        super(LNAutoEncoder, self).__init__(model_name)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, padding=3),
            nn.LayerNorm([16, 128, 128]),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.LayerNorm([32, 64, 64]),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.LayerNorm([64, 16, 16]),
            nn.GELU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.LayerNorm([32, 32, 32]),
            nn.GELU(),
            nn.ConvTranspose2d(
                32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.LayerNorm([16, 64, 64]),
            nn.GELU(),
            nn.ConvTranspose2d(
                16, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x