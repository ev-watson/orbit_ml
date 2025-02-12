import joblib
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

import config


class Scaler:
    def __init__(self):
        self.mean = None  # Mean per feature
        self.std = None  # Standard deviation per feature
        self.axes = None  # Axes along which mean and std were computed
        self.is_fitted = False  # Flag to check if scaler has been fitted

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fits the scaler to the data and transforms it.
        :param data: np.ndarray, input data of shape [N, F] or [N, S, F]
        :return: np.ndarray, scaled data with the same shape as input
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy array.")

        if data.ndim == 3:
            self.axes = (0, 1)  # Compute mean/std over N and S
        elif data.ndim == 2:
            self.axes = (0,)  # Compute mean/std over N
        else:
            raise ValueError("Data must be either 2D [N, F] or 3D [N, S, F].")

        self.mean = data.mean(axis=self.axes)
        self.std = data.std(axis=self.axes) + 1e-12  # Add epsilon to avoid division by zero

        if data.ndim == 3:
            mean_reshaped = self.mean.reshape(1, 1, -1)  # Shape: [1, 1, F]
            std_reshaped = self.std.reshape(1, 1, -1)  # Shape: [1, 1, F]
        elif data.ndim == 2:
            mean_reshaped = self.mean.reshape(1, -1)  # Shape: [1, F]
            std_reshaped = self.std.reshape(1, -1)  # Shape: [1, F]

        scaled_data = (data - mean_reshaped) / std_reshaped
        self.is_fitted = True
        return scaled_data

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transforms a PyTorch tensor using the fitted scaler.
        :param tensor: torch.Tensor, input tensor of shape [N, F] or [N, S, F]
        :return: torch.Tensor, scaled tensor with the same shape as input
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")

        if tensor.ndim == 3 and self.axes != (0, 1):
            raise ValueError("Scaler was fitted on 2D data, but received 3D tensor.")
        if tensor.ndim == 2 and self.axes != (0,):
            raise ValueError("Scaler was fitted on 3D data, but received 2D tensor.")

        mean = torch.from_numpy(self.mean).to(tensor.device)
        std = torch.from_numpy(self.std).to(tensor.device)

        if tensor.ndim == 3:
            mean = mean.view(1, 1, -1)  # Shape: [1, 1, F]
            std = std.view(1, 1, -1)  # Shape: [1, 1, F]
        elif tensor.ndim == 2:
            mean = mean.view(1, -1)  # Shape: [1, F]
            std = std.view(1, -1)  # Shape: [1, F]

        scaled_tensor = (tensor - mean) / std
        return scaled_tensor

    def inverse_transform(self, scaled_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverts the scaling of a PyTorch tensor using the fitted scaler.
        :param scaled_tensor: torch.Tensor, scaled tensor of shape [N, F] or [N, S, F]
        :return: torch.Tensor, original tensor before scaling
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler has not been fitted yet.")

        if not isinstance(scaled_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")

        if scaled_tensor.ndim == 3 and self.axes != (0, 1):
            raise ValueError("Scaler was fitted on 2D data, but received 3D tensor.")
        if scaled_tensor.ndim == 2 and self.axes != (0,):
            raise ValueError("Scaler was fitted on 3D data, but received 2D tensor.")

        mean = torch.from_numpy(self.mean).to(scaled_tensor.device)
        std = torch.from_numpy(self.std).to(scaled_tensor.device)

        if scaled_tensor.ndim == 3:
            mean = mean.view(1, 1, -1)  # Shape: [1, 1, F]
            std = std.view(1, 1, -1)  # Shape: [1, 1, F]
        elif scaled_tensor.ndim == 2:
            mean = mean.view(1, -1)  # Shape: [1, F]
            std = std.view(1, -1)  # Shape: [1, F]

        # Revert the scaling
        original_tensor = scaled_tensor * std + mean
        return original_tensor


class SEBlock(LightningModule):
    def __init__(self, channel, reduction=config.SE_REDUCTION):
        super(SEBlock, self).__init__()
        self.se_block = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.ndim == 3:
            y = torch.mean(x, dim=(0, 1))  # (channel,)
        elif x.ndim == 2:
            y = torch.mean(x, dim=0)  # (channel,)
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
        :param X: torch.Tensor, list of value pairs to use as input (excluding target), must be raw numbers unscaled.
        :return: Tensor of predicted target values.
        """
        self.eval()
        device = next(self.parameters()).device
        if config.SCALE:
            scalers = joblib.load(config.SCALER_FILE)
            input_scaler = scalers['input_scaler']
            target_scaler = scalers['target_scaler']
            input_data = input_scaler.transform(X.to(dtype=torch.get_default_dtype(), device=device))
            with torch.no_grad():
                output_scaled = self.forward(input_data)
            out = target_scaler.inverse_transform(output_scaled)
        else:
            with torch.no_grad():
                out = self.forward(X.to(dtype=torch.get_default_dtype(), device=device))
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
