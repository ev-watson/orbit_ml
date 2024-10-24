import joblib
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

import config


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        if data.ndim != 3:
            raise ValueError("Data must be a 3D array with shape (N, S, F).")

        self.mean = data.mean(axis=(0, 1))
        self.std = data.std(axis=(0, 1)) + 1e-12

        mean_reshaped = self.mean.reshape(1, 1, -1)
        std_reshaped = self.std.reshape(1, 1, -1)

        scaled_data = (data - mean_reshaped) / std_reshaped
        return scaled_data

    def transform(self, tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 3:
            raise ValueError("Input tensor must be 3D with shape [N, S, F].")

        mean = torch.from_numpy(self.mean).to(tensor.device)
        std = torch.from_numpy(self.std).to(tensor.device)
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)

        scaled_tensor = (tensor - mean) / std
        return scaled_tensor

    def inverse_transform(self, scaled_tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        if not isinstance(scaled_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if scaled_tensor.ndim != 3:
            raise ValueError("Input tensor must be 3D with shape [N, S, F].")

        mean = torch.from_numpy(self.mean).to(scaled_tensor.device)
        std = torch.from_numpy(self.std).to(scaled_tensor.device)
        mean = mean.view(1, 1, -1)
        std = std.view(1, 1, -1)

        original_tensor = scaled_tensor * std + mean
        return original_tensor


class SEBlock(LightningModule):
    def __init__(self, channel, reduction=config.REDUCTION):
        super(SEBlock, self).__init__()
        self.se_block = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = torch.mean(x, dim=(0, 1))  # (channel,)
        y = self.se_block(y)
        y = y.unsqueeze(0)  # (1, channel)
        return x * y, y.squeeze(0)  # (batch_size, channel), (channel,)


class PredictorMixin:
    """
    Generic prediction mixin for use with unscaled inputs
    If no scaling this function is no different than calling self.forward with eval and no_grad)
    """
    input_slice = None
    output_slice = None

    def predict(self, X):
        """
        Prediction method that uses the same scaling function as training for use after model training.
        Must have scaler used with the original dataset saved as pkl file in cwd with name specified in config.
        :param X: array-like, list of value pairs to use as input (excluding target), must be raw numbers unscaled.
        :return: array of predicted target values.
        """
        self.eval()
        device = next(self.parameters()).device
        if config.SCALE:
            scalers = joblib.load(config.SCALER_FILE)
            input_scaler = scalers['input_scaler']
            target_scaler = scalers['target_scaler']
            input_data = input_scaler.transform(torch.from_numpy(X).to(dtype=torch.get_default_dtype(), device=device))
            with torch.no_grad():
                output_scaled = self.forward(input_data)
            out = target_scaler.inverse_transform(output_scaled)
        else:
            with torch.no_grad():
                out = self.forward(torch.from_numpy(X).to(dtype=torch.get_default_dtype(), device=device))
        return out


class GradientNormCallback(Callback):
    def __init__(self):
        """
        Initializes the GradientNormCallback.
        """
        super().__init__()

    def on_after_backward(self, trainer, pl_module):
        if trainer.training:
            total_norm = 0.0
            for param in pl_module.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            pl_module.log('grad_norm', total_norm, prog_bar=True, logger=True, sync_dist=True, on_epoch=True,
                          on_step=config.ON_STEP)
