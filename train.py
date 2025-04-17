import torch.optim as optim
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from data_construction import *

seed = config.SEED if config.SEED else np.random.randint(1, 10000)
print_block(f"SEED: {seed}")
seed_everything(seed)

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

params = {
    'lr': 0.00045830946730129885,
    'hidden_dim': 2048,
    'num_layers': 5,
    'se_block': True if not config.MAC else False,      # mac's MPS can't handle the SE Block
    'se_reduction': 64,
    'rotational_equivariance': True,
    'windowed': False,
    'drop_rate': 0.14152253394728245,
    'dropout_frequency': 3,
    'loss': F.l1_loss,
    # 'loss_kwargs': {
    #     'beta': 1.6149167925554124,
    # },
    'activation': F.relu,
    'optimizer': optim.NAdam,
    'optimizer_kwargs': {
        'betas': (0.9105880024486549, 0.9915574031641383),
        'weight_decay': 3.7148597273862825e-09,
        'eps': 1e-8,
        'momentum_decay': 0.03288402439883893,
        'decoupled_weight_decay': True,
    },
    # 'scheduler': optim.lr_scheduler.CyclicLR,
    # 'scheduler_kwargs': {
    #     'base_lr': 7e-4,
    #     'max_lr': .01,
    #     'step_size_up': 2000,
    #     'scale_fn': None,
    #     'mode': 'triangular',   # only used if 'scale_fn' is None
    #     'gamma': 1.0,   # only used if 'mode' = 'exp_range'
    # },
    'scheduler_kwargs': {
        'factor': 0.06603369129970485,
        'patience': 4,
    },
}

config.ON_STEP = False
config.WINDOWED = params['windowed']
config.ROTATIONAL_EQUIVARIANCE = params['rotational_equivariance']
# config.SEQUENCE_LENGTH = 200

config.update_hparams(params)

data_module = NNDataModule(batch_size=16)

model = GNN(
    **params,
)

trainer = Trainer(
    max_epochs=5,
    callbacks=[EarlyStopping(monitor='val_loss', patience=config.PATIENCE, mode='min'),
               GradientNormCallback(),
               LearningRateMonitor(logging_interval='step' if config.ON_STEP else 'epoch'),
               ],
    gradient_clip_val=config.GRADIENT_CLIP_VAL,
    accelerator='gpu',
    devices=-1,
    strategy='auto',
    sync_batchnorm=True,
    logger=TensorBoardLogger('tlogs', name=f"final"),
)

trainer.fit(model, datamodule=data_module)

trainer.test(model, datamodule=data_module)
