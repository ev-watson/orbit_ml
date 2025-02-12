import numpy as np
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn

from utils import *

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)


class BaseArch(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.lr = kwargs.get('lr', config.LEARNING_RATE)
        self.activation = kwargs.get('activation', F.hardswish)
        self.loss = kwargs.get('loss', F.l1_loss)
        self.optimizer = kwargs.get('optimizer', torch.optim.AdamW)
        self.scheduler = kwargs.get('scheduler', torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.loss_kwargs = {}
        self.loss_kwargs.update(kwargs.get('loss_kwargs', {}))
        self.optimizer_kwargs = {'params': self.parameters(), 'lr': self.lr, 'weight_decay': config.WEIGHT_DECAY}
        self.optimizer_kwargs.update(kwargs.get('optimizer_kwargs', {}))
        self.scheduler_kwargs = {}
        self.scheduler_kwargs.update(kwargs.get('scheduler_kwargs', {}))

        self.save_hyperparameters(config.hparams)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view_as(y_hat), **self.loss_kwargs)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=config.ON_STEP)

        if config.LOG_ATTN and hasattr(self, 'attns') and self.attns is not None:
            for i, attn in np.ndenumerate(self.attns):
                self.log(f'attention_feature_{i}', attn, sync_dist=True, prog_bar=False, logger=True, on_epoch=True,
                         on_step=config.ON_STEP)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view_as(y_hat), **self.loss_kwargs)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.view_as(y_hat), **self.loss_kwargs)
        self.log('test_loss', loss, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(**self.optimizer_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        if self.scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.s_config = {"scheduler": scheduler, "monitor": "val_loss"}
        elif self.scheduler == torch.optim.lr_scheduler.OneCycleLR or self.scheduler == torch.optim.lr_scheduler.CyclicLR:
            self.s_config = {'scheduler': scheduler, 'interval': 'step'}
        return {"optimizer": optimizer, "lr_scheduler": self.s_config}


class MLP(BaseArch):
    input_slice = slice(None, 6)
    output_slice = slice(6, 7)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = kwargs.get('input_dim', 6)
        self.output_dim = kwargs.get('output_dim', 1)
        self.hidden_dim = kwargs.get('hidden_dim', config.HIDDEN_DIM)
        self.num_layers = kwargs.get('num_layers', config.NUM_LAYERS)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.mlp_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.mlp_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


@config.register('gnn')
class GNN(BaseArch, PredictorMixin):
    input_slice = slice(1, 7)
    output_slice = slice(7, 8)
    input_dim = len(range(*input_slice.indices(10)))
    output_dim = len(range(*output_slice.indices(10)))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = kwargs.get('input_dim', self.input_dim)
        self.output_dim = kwargs.get('output_dim', self.output_dim)
        self.hidden_dim = kwargs.get('hidden_dim', config.HIDDEN_DIM)
        self.num_layers = kwargs.get('num_layers', config.NUM_LAYERS)
        self.dropout_frequency = kwargs.get('dropout_frequency', config.DROPOUT_FREQUENCY)
        self.drop_rate = kwargs.get('drop_rate', config.DROP_RATE)
        self.use_se = kwargs.get('se_block', config.USE_SE)
        self.se_reduction = kwargs.get('se_reduction', config.SE_REDUCTION)

        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.mlp_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout_input = nn.Dropout(p=self.drop_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(p=self.drop_rate) for _ in range(self.num_layers)])
        if self.use_se:
            self.attns = None
            self.se_input = SEBlock(self.input_dim, reduction=self.se_reduction)
            self.se_block = nn.ModuleList([SEBlock(self.hidden_dim, reduction=self.se_reduction) for _ in range(self.num_layers)])

        self.save_hyperparameters()

        # self.m = 1.652e-7    # mass of merc in M_solar
        # self.m = 0.3301e24   # mass of merc in kg
        # self.mass_merc = nn.Parameter(torch.tensor([0.05], dtype=torch.get_default_dtype()), requires_grad=True)

    def forward(self, x):
        if self.use_se:
            x, attn = self.se_input(x)
            self.attns = attn.detach().cpu().numpy()
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout_input(x)
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i % self.dropout_frequency == 0:
                x = self.dropouts[i](x)
            else:
                x = self.activation(x)
            if self.use_se:
                x, attn = self.se_block[i](x)
        x = self.output_layer(x)
        return x

    # def forward(self, merc_eph):
    #     param_device = next(self.parameters()).device
    #     x = merc_eph.to(param_device).clone()
    #
    #     batch_size = x.size(0)
    #     solmass = torch.ones(batch_size, 1, device=param_device)
    #     mercmass = torch.clamp(self.mass_merc.expand(batch_size, 1), 0, 0.5)
    #
    #     inputs = torch.cat([x, solmass, mercmass], dim=1)
    #     force_vec = self.mlp(inputs)
    #     acceleration = force_vec / self.mass_merc
    #
    #     return acceleration


@config.register('interp')
class InterpMLP(MLP, PredictorMixin):
    input_slice = slice(None, -1)
    output_slice = -1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()


class CNN(BaseArch):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.conv_layers = nn.ModuleList()

        for _ in range(num_layers):
            conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv_layers.append(conv_layer)
            input_dim = hidden_dim  # Update input_dim to channel for the next layer

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # move column 2 to index 1 and column 1 to index 2
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        x = x[:, :, -1]  # takes last of [batch_size, channel, new_sequence_length] to make it [batch_size, channel]
        x = self.fc(x)
        return x


class LSTM(BaseArch):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, lr=6e-5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.activation(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x
