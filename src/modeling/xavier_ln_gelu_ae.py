import torch
import torch.nn as nn

from src.modeling.base import BaseModel


class XAvierLNAutoEncoder(BaseModel):
    def __init__(self, model_name='xv_ln_ae'):
        super(XAvierLNAutoEncoder, self).__init__(model_name)
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
            nn.AvgPool2d(2),
        )

        self.decoder = nn.Sequential(
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