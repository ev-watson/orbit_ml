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


@config.register('semlp')
class SEMLP(BaseArch, PredictorMixin):
    """
    Legacy 'GNN' from the first iteration of this project: an MLP with squeeze-and-excitation
    channel gating between layers, fed flat per-timestep features. Kept here as a baseline only;
    it is not a graph network and does not perform message passing.
    """
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

    def forward(self, x):
        if self.use_se:
            x, attn = self.se_input(x)
            self.attns = attn.detach().cpu().numpy()
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout_input(x)
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            x = self.activation(x)
            if self.use_se:
                x, attn = self.se_block[i](x)
            if (i+1) % self.dropout_frequency == 0:
                x = self.dropouts[i](x)
        x = self.output_layer(x)
        return x


@config.register('gnn')
class MPNN(BaseArch, PredictorMixin):
    """
    Cranmer-style message-passing graph neural network for orbital dynamics.

    Each input is a graph snapshot of B bodies with per-node features
    ``[mass, x, y, z, vx, vy, vz]``; the per-node output is the body's acceleration
    ``[ax, ay, az]``. The forward pass evaluates a learned edge function ``phi_e`` on every ordered
    pair (i, j) to produce a low-dimensional message ``m_ij``, sums incoming messages at each
    destination node, then applies a learned node function ``phi_v`` to predict the per-node
    target. An L1 regularization term on messages (``msg_l1``) encourages a sparse, interpretable
    bottleneck so that each active message channel can later be distilled to a closed-form
    expression by symbolic regression (the Cranmer recipe).

    This module assumes a fixed graph topology shared across the batch — the dataset emits
    identical edge_index / dst_index tensors per snapshot — which is appropriate for fully
    connected N-body problems with constant body count.
    """
    # Per-node feature slicing on the [F = 10] last axis of each snapshot tensor.
    input_slice = slice(0, 7)       # mass, x, y, z, vx, vy, vz
    output_slice = slice(7, 10)     # ax, ay, az
    input_dim = len(range(*input_slice.indices(10)))
    output_dim = len(range(*output_slice.indices(10)))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_input_dim = kwargs.get('node_input_dim', self.input_dim)
        self.node_output_dim = kwargs.get('node_output_dim', self.output_dim)
        self.edge_hidden_dim = kwargs.get('edge_hidden_dim', config.EDGE_HIDDEN_DIM)
        self.edge_layers = kwargs.get('edge_layers', config.EDGE_LAYERS)
        self.node_hidden_dim = kwargs.get('node_hidden_dim', config.NODE_HIDDEN_DIM)
        self.node_layers = kwargs.get('node_layers', config.NODE_LAYERS)
        self.msg_dim = kwargs.get('msg_dim', config.MSG_DIM)
        self.msg_l1 = kwargs.get('msg_l1', config.MSG_L1)
        self.drop_rate = kwargs.get('drop_rate', config.DROP_RATE)
        self.dropout_frequency = kwargs.get('dropout_frequency', config.DROPOUT_FREQUENCY)
        self.use_se = kwargs.get('se_block', config.USE_SE)
        self.se_reduction = kwargs.get('se_reduction', config.SE_REDUCTION)

        self.edge_model = MLPBlock(
            input_dim=2 * self.node_input_dim,
            hidden_dim=self.edge_hidden_dim,
            output_dim=self.msg_dim,
            num_layers=self.edge_layers,
            activation=self.activation,
            drop_rate=self.drop_rate,
            dropout_frequency=self.dropout_frequency,
            use_se=self.use_se,
            se_reduction=self.se_reduction,
        )
        self.node_model = MLPBlock(
            input_dim=self.node_input_dim + self.msg_dim,
            hidden_dim=self.node_hidden_dim,
            output_dim=self.node_output_dim,
            num_layers=self.node_layers,
            activation=self.activation,
            drop_rate=self.drop_rate,
            dropout_frequency=self.dropout_frequency,
            use_se=self.use_se,
            se_reduction=self.se_reduction,
        )

        self.last_messages = None       # cached for L1 reg and for symbolic distillation
        self.attns = None               # passthrough so BaseArch logging hook stays happy

        self.save_hyperparameters()

    def forward(self, batch):
        """
        :param batch: dict with keys
            - 'nodes':      tensor (B, N, F_in)        per-node input features
            - 'src_index':  tensor (M,) long           source node index of each directed edge
            - 'dst_index':  tensor (M,) long           destination node index of each directed edge
            - 'predict_mask' (optional): tensor (N,) bool indicating which nodes contribute to loss
        :return: tensor (B, N, F_out), per-node predictions (e.g. accelerations).
        """
        V = batch['nodes']
        src_index = batch['src_index']
        dst_index = batch['dst_index']
        N = V.shape[1]

        V_src = V.index_select(1, src_index)        # (B, M, F_in)
        V_dst = V.index_select(1, dst_index)        # (B, M, F_in)
        edge_input = torch.cat([V_src, V_dst], dim=-1)
        messages = self.edge_model(edge_input)      # (B, M, msg_dim)
        self.last_messages = messages

        agg = scatter_sum(messages, dst_index, num_nodes=N)  # (B, N, msg_dim)
        node_input = torch.cat([V, agg], dim=-1)
        return self.node_model(node_input)          # (B, N, F_out)

    def _masked_loss(self, y_hat, y, mask):
        """
        Mean per-target-component loss restricted to the predict-masked nodes.
        """
        if mask is not None:
            y_hat = y_hat[:, mask, :]
            y = y[:, mask, :]
        return self.loss(y_hat, y, **self.loss_kwargs)

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._masked_loss(y_hat, y.view_as(y_hat), x.get('predict_mask'))
        if self.msg_l1 and stage == 'train' and self.last_messages is not None:
            loss = loss + self.msg_l1 * self.last_messages.abs().mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'train')
        self.log('train_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True,
                 on_step=config.ON_STEP)
        if config.LOG_ATTN and self.last_messages is not None:
            self.log('msg_abs_mean', self.last_messages.abs().mean(),
                     sync_dist=True, logger=True, on_epoch=True, on_step=config.ON_STEP)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'val')
        self.log('val_loss', loss, sync_dist=True, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, 'test')
        self.log('test_loss', loss, sync_dist=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def predict(self, batch):
        """
        Prediction method for raw, unscaled graph batches. This mirrors ``PredictorMixin`` for
        the graph-specific forward signature: ``batch`` must be a dict with raw physical-unit
        ``nodes`` plus the usual edge-index tensors. The returned acceleration is inverse-scaled
        back to physical units when training used scalers.

        :param batch: dict accepted by :meth:`forward`, with unscaled ``nodes``.
        :return: tensor (B, N, F_out), predictions in target physical units.
        """
        self.eval()
        device = next(self.parameters()).device
        model_batch = dict(batch)
        model_batch['nodes'] = batch['nodes'].to(dtype=torch.get_default_dtype(), device=device)
        model_batch['src_index'] = batch['src_index'].to(device=device)
        model_batch['dst_index'] = batch['dst_index'].to(device=device)
        if batch.get('predict_mask') is not None:
            model_batch['predict_mask'] = batch['predict_mask'].to(device=device)

        scalers = ScalerBundle.maybe_load()
        if scalers is not None:
            model_batch = scalers.scale_graph_batch(model_batch)

        out = self.forward(model_batch)
        if scalers is not None:
            out = scalers.unscale_targets(out)
        return out

    @torch.no_grad()
    def edge_messages(self, batch):
        """
        Run the edge function only and return raw messages alongside the (V_src, V_dst) inputs.
        Used by :mod:`pysr_main` to harvest distillation targets for the learned force law.

        :param batch: same dict layout as :meth:`forward`.
        :return: tuple (edge_input, messages) of np.ndarrays. ``edge_input`` is shape
            (B, M, 2 * F_in); ``messages`` is shape (B, M, msg_dim).
        """
        self.eval()
        V = batch['nodes']
        V_src = V.index_select(1, batch['src_index'])
        V_dst = V.index_select(1, batch['dst_index'])
        edge_input = torch.cat([V_src, V_dst], dim=-1)
        messages = self.edge_model(edge_input)
        return edge_input.cpu().numpy(), messages.cpu().numpy()


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
