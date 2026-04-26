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
    'lr': 5e-4,
    # phi_e: edge function
    'edge_hidden_dim': 256,
    'edge_layers': 3,
    # phi_v: node function
    'node_hidden_dim': 256,
    'node_layers': 3,
    # message bottleneck (Cranmer): keep small for cleaner symbolic distillation
    'msg_dim': 100,
    'msg_l1': 1e-2,
    # regularization carried over from the SEMLP baseline
    'drop_rate': 0.1,
    'dropout_frequency': 3,
    'se_block': True if not config.MAC else False,      # mac's MPS can't handle the SE Block
    'se_reduction': 16,
    'loss': F.l1_loss,
    'activation': F.hardswish,
    'optimizer': optim.NAdam,
    'optimizer_kwargs': {
        'betas': (0.9, 0.999),
        'weight_decay': 1e-8,
        'eps': 1e-8,
        'momentum_decay': 4e-3,
        'decoupled_weight_decay': True,
    },
    'scheduler_kwargs': {
        'factor': 0.1,
        'patience': 4,
    },
}

config.ON_STEP = False
# WINDOWED is meaningless for the graph network -- each snapshot is its own graph.
config.WINDOWED = False

config.update_hparams(params)

data_module = NNDataModule(batch_size=16)

model = config.retrieve('model')(**params)

trainer = Trainer(
    max_epochs=25,
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
