import os

import torch
import torch.nn as nn
from loguru import logger


class BaseModel(nn.Module):
    def __init__(self, model_name="base_model"):
        super(BaseModel, self).__init__()
        self.model_name = model_name

    def save(self, directory="models"):
        os.makedirs(directory, exist_ok=True)

        path = os.path.join(directory, f"{self.model_name}.pth")
        torch.save(self.state_dict(), path)
        logger.info(f"Модель сохранена в {path}")

    def load(self, device, directory="models"):
        path = os.path.join(directory, f"{self.model_name}.pth")
        self.load_state_dict(torch.load(path, map_location=device))
        logger.info(f"Модель загружена из {path}")
