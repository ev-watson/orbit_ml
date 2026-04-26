import joblib
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

import config


class Scaler:
    """
    Standard scaler, removes mean and scales to unit variance
    """
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

        orig_dtype = data.dtype
        data64 = data.astype(np.float64, copy=False)

        if data64.ndim == 3:
            self.axes = (0, 1)  # Compute mean/std over N and S
        elif data64.ndim == 2:
            self.axes = (0,)  # Compute mean/std over N
        else:
            raise ValueError("Data must be either 2D [N, F] or 3D [N, S, F].")

        self.mean = data64.mean(axis=self.axes)
        self.std = data64.std(axis=self.axes) + 1e-12  # Add epsilon to avoid division by zero

        if data64.ndim == 3:
            mean_reshaped = self.mean.reshape(1, 1, -1)  # Shape: [1, 1, F]
            std_reshaped = self.std.reshape(1, 1, -1)  # Shape: [1, 1, F]
        elif data64.ndim == 2:
            mean_reshaped = self.mean.reshape(1, -1)  # Shape: [1, F]
            std_reshaped = self.std.reshape(1, -1)  # Shape: [1, F]

        scaled_data = (data64 - mean_reshaped) / std_reshaped
        self.is_fitted = True
        return scaled_data.astype(orig_dtype, copy=False)

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

        # change to default dtype so non-mac scaler pickles can be compatible with mac
        return scaled_tensor.to(dtype=torch.get_default_dtype())

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


class ScalerBundle:
    """
    Small wrapper around the paired input/target scalers saved during training.

    Keeping this wrapper as the only place that knows the on-disk pickle format avoids the
    previous pattern of callers loading ``{'input_scaler': ..., 'target_scaler': ...}``
    directly and hand-scaling in slightly different ways.
    """
    def __init__(self, input_scaler, target_scaler):
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

    @classmethod
    def load(cls, path=None):
        path = path if path is not None else config.SCALER_FILE
        raw = joblib.load(path)
        if isinstance(raw, cls):
            return raw
        return cls(raw['input_scaler'], raw['target_scaler'])

    @classmethod
    def maybe_load(cls, path=None):
        if not config.SCALE:
            return None
        return cls.load(path)

    def dump(self, path=None):
        path = path if path is not None else config.SCALER_FILE
        joblib.dump({
            'input_scaler': self.input_scaler,
            'target_scaler': self.target_scaler,
        }, path)

    def scale_inputs(self, inputs):
        return self.input_scaler.transform(inputs)

    def unscale_targets(self, targets):
        return self.target_scaler.inverse_transform(targets)

    def scale_graph_batch(self, batch):
        scaled = dict(batch)
        scaled['nodes'] = self.scale_inputs(batch['nodes'])
        return scaled


class SEBlock(LightningModule):
    def __init__(self, channel, reduction=config.SE_REDUCTION):
        super(SEBlock, self).__init__()
        bottleneck = max(channel // reduction, 1)
        self.se_block = nn.Sequential(
            nn.Linear(channel, bottleneck, bias=False),
            nn.ReLU(),
            nn.Linear(bottleneck, channel, bias=False),
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


class MLPBlock(nn.Module):
    """
    Stacked-linear block used as the backbone of the message-passing GNN's edge and node functions.
    Operates on the trailing feature axis, so the same module handles tensors shaped (B, F),
    (B, N, F), or (B, M, F) interchangeably.

    :param input_dim: int, number of input features.
    :param hidden_dim: int, hidden width.
    :param output_dim: int, output features.
    :param num_layers: int, number of hidden Linear layers (excluding input/output projections).
    :param activation: callable, activation function applied between layers.
    :param drop_rate: float, dropout probability.
    :param dropout_frequency: int, every X layers dropout is applied (smaller value = more dropout).
    :param use_se: bool, whether to apply SE-style channel gating after every hidden layer.
    :param se_reduction: int, SE bottleneck reduction factor.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers=2,
                 activation=None,
                 drop_rate=0.0,
                 dropout_frequency=1,
                 use_se=False,
                 se_reduction=config.SE_REDUCTION):
        super().__init__()
        self.activation = activation if activation is not None else nn.functional.hardswish
        self.use_se = use_se
        self.dropout_frequency = max(int(dropout_frequency), 1)

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout_input = nn.Dropout(p=drop_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(p=drop_rate) for _ in range(num_layers)])
        if self.use_se:
            self.se_input = SEBlock(input_dim, reduction=se_reduction)
            self.se_hidden = nn.ModuleList(
                [SEBlock(hidden_dim, reduction=se_reduction) for _ in range(num_layers)]
            )
        self.attns = None  # last-channel-attention for logging

    def forward(self, x):
        if self.use_se:
            x, attn_in = self.se_input(x)
            self.attns = attn_in.detach().cpu().numpy()
        x = self.activation(self.input_layer(x))
        x = self.dropout_input(x)
        for i, layer in enumerate(self.hidden_layers):
            x = self.activation(layer(x))
            if self.use_se:
                x, _ = self.se_hidden[i](x)
            if (i + 1) % self.dropout_frequency == 0:
                x = self.dropouts[i](x)
        return self.output_layer(x)


def scatter_sum(messages, dst_index, num_nodes):
    """
    Sum messages into per-destination-node aggregates.

    :param messages: torch.Tensor, shape (B, M, F): edge messages, B batches, M edges, F features.
    :param dst_index: torch.LongTensor, shape (M,): destination node index for each edge,
        shared across the batch (graph topology is identical batch-to-batch).
    :param num_nodes: int, number of nodes per snapshot.
    :return: torch.Tensor, shape (B, num_nodes, F): aggregated messages per destination node.
    """
    B, M, F = messages.shape
    out = messages.new_zeros((B, num_nodes, F))
    idx = dst_index.view(1, M, 1).expand(B, M, F)
    out.scatter_add_(1, idx, messages)
    return out


def fully_connected_edges(num_bodies, device=None):
    """
    Build directed edge-index tensors for a fully connected graph with no self-loops.
    """
    src, dst = [], []
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                src.append(i)
                dst.append(j)
    return (torch.tensor(src, dtype=torch.long, device=device),
            torch.tensor(dst, dtype=torch.long, device=device))


class PredictorMixin:
    """
    Generic prediction mixin for use with unscaled inputs on a model that was trained with scaled inputs
    If no scaling was involved this function is no different than calling self.forward with eval and no_grad)
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
        input_data = X.to(dtype=torch.get_default_dtype(), device=device)
        scalers = ScalerBundle.maybe_load()
        if scalers is not None:
            input_data = scalers.scale_inputs(input_data)
        with torch.no_grad():
            out = self.forward(input_data)
        if scalers is not None:
            out = scalers.unscale_targets(out)
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
