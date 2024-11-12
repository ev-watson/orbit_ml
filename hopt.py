import argparse
import signal

import optuna
import torch_optimizer as optim
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lion_pytorch import Lion

from data_construction import *

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

activation_functions = {
    'relu': F.relu,
    'gelu': F.gelu,
    'tanh': F.tanh,
    'mish': F.mish,
    'hardswish': F.hardswish,
    'sigmoid': F.sigmoid,
    # 'swish': lambda x: x * F.sigmoid(x),
    'sinu': lambda x: x + torch.sin(x) ** 2,
}

loss_functions = {
    'l1': F.l1_loss,
    'smooth_l1': F.smooth_l1_loss,
    'huber': F.huber_loss,
    'mse': F.mse_loss,
    'rmwe': rmwe_loss,
}

optimizer_functions = {
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW,
    'nadam': torch.optim.NAdam,
    'radam': torch.optim.RAdam,
    'adabound': optim.AdaBound,
    'swats': optim.SWATS,
    'lion': Lion,
}

base_opt_kwargs = {
    'betas1': {'type': 'float', 'low': 0.9, 'high': 0.99},  # Log inherently included in sampler function in utils
    'betas2': {'type': 'float', 'low': 0.99, 'high': 0.9999},
    'eps': {'type': 'float', 'default': 1e-10},
    'weight_decay': {'type': 'float', 'low': 1e-8, 'high': 1e-2, 'log': True},
}

optimizer_hyperparams = {
    'sgd': {
        'momentum': {'type': 'float', 'low': 0.8, 'high': 0.99999},
        'weight_decay': {'type': 'float', 'low': 1e-8, 'high': 1e-2, 'log': True},
        'nesterov': {'type': 'bool', 'default': True},
    },
    'adamw': {
        **base_opt_kwargs,
    },
    'nadam': {
        **base_opt_kwargs,
        'momentum_decay': {'type': 'float', 'low': 1e-6, 'high': 1e-1, 'log': True},
        'decoupled_weight_decay': {'type': 'bool', 'default': True},
    },
    'radam': {
        **base_opt_kwargs,
        'decoupled_weight_decay': {'type': 'bool', 'default': True},
    },
    'adabound': {
        **base_opt_kwargs,
        'final_lr': {'type': 'float', 'low': 1e-8, 'high': 1e-1},
        'gamma': {'type': 'float', 'low': 1e-6, 'high': 1e-1},
        'amsbound': {'type': 'bool'},
    },
    'swats': {
        **base_opt_kwargs,
        'amsgrad': {'type': 'bool'},
        'nesterov': {'type': 'bool'},
    },
    'lion': {
        **{k: v for k, v in base_opt_kwargs.items() if k != 'eps'},
        'decoupled_weight_decay': {'type': 'bool'},
    },
}

loss_hyperparams = {
    'huber': {
        'delta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
    'smooth_l1': {
        'beta': {'type': 'float', 'low': 1e-1, 'high': 2e0}
    },
}

parser = argparse.ArgumentParser(description="Hyper-optimization")
optimizer_choices = list(optimizer_functions.keys())
model_choices = ['interp', 'gnn']
parser.add_argument("--opt", "-o", type=str, default="adamw",
                    choices=optimizer_choices,
                    help=f"Optimizer function. Defaults to adamw")
parser.add_argument("--model", '-m', type=str, default='gnn',
                    choices=model_choices,
                    help=f"Model to run. Defaults to gnn")
args = parser.parse_args()
config.TYPE = args.model


def objective(trial):
    seed_everything(config.SEED)
    clear_local_ckpt_files()

    params = {
        'lr': trial.suggest_float('lr', 1e-7, 1e-2, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256, 512, 1024]),
        'num_layers': trial.suggest_int('num_layers', 2, 8),
        # 'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32]),
        'drop_rate': trial.suggest_float('drop_rate', 0.01, 0.5),
        'se_block': trial.suggest_categorical('se_block', [True, False]),
        # 'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
        'rotational_equivariance': trial.suggest_categorical('rotational_equivariance', [True, False]),
        'windowed': trial.suggest_categorical('windowed', [True, False]),
        'gradient_clip_val': trial.suggest_float('gradient_clip_val', 0.7, 1.5),
        # 'activation_name': 'hardswish',
        'activation_name': trial.suggest_categorical('activation', list(activation_functions.keys())),
        # 'loss_name': 'l1',
        'loss_name': trial.suggest_categorical('loss_name', list(loss_functions.keys())),
        'optimizer': optimizer_functions[args.opt],
    }
    config.ROTATIONAL_EQUIVARIANCE = params.get('rotational_equivariance', False)
    config.WINDOWED = params.get('windowed', False)
    if config.WINDOWED:
        params['seqlen'] = trial.suggest_categorical('seqlen', [50, 100, 150, 200, 250, 300])
        config.SEQUENCE_LENGTH = params['seqlen']

    params['loss'] = loss_functions[params['loss_name']]
    params['scheduler_kwargs'] = {
        'factor': trial.suggest_float('scheduler_factor', 0.05, .5),
        'patience': 3,
    }

    data_module = NNDataModule()

    if args.opt in optimizer_hyperparams:
        optimizer_params = sample_hyperparams(trial, optimizer_hyperparams[args.opt])
    else:
        raise KeyError(f"{args.opt} not in registered optimizers")
    params['optimizer_kwargs'] = optimizer_params

    loss_params = {}
    if params['loss_name'] in loss_hyperparams:
        loss_params = sample_hyperparams(trial, loss_hyperparams[params['loss_name']])
    params['loss_kwargs'] = loss_params

    config.update_hparams(params)

    model = config.retrieve('model')(
        **params
    )

    print_err(f"Starting trial with parameters: {params}")

    trainer = Trainer(
        max_epochs=config.MAX_EPOCHS,
        gradient_clip_val=params.get('gradient_clip_val', config.GRADIENT_CLIP_VAL),
        callbacks=[EarlyStopping(monitor='val_loss', patience=config.PATIENCE, mode='min'),
                   GradientNormCallback()],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)] if not config.MAC else None,
        accelerator='gpu',
        devices=-1,
        strategy='ddp' if not config.MAC else 'auto',
        sync_batchnorm=True,
        benchmark=True,
        logger=TensorBoardLogger('hopt', name=f'{args.model}_{args.opt}_logs'),
    )

    trainer.fit(model, datamodule=data_module)

    rtrials = 50000
    if args.model == 'interp':
        mae, mape = interp_test(model, ntrials=rtrials, mape=True, err=True)
    else:
        mae, mape = gnn_test(model, ntrials=rtrials, mape=True, err=True, mean_axis=None)

    return mae.item()


sampler = optuna.samplers.NSGAIISampler(
    population_size=50,
    seed=config.SEED,
)
study_name = f"{args.model}_{args.opt}_study"
storage_name = f"sqlite:///{study_name}.db"
study = optuna.create_study(direction='minimize',
                            storage=storage_name,
                            sampler=sampler,
                            study_name=study_name,
                            load_if_exists=True)
study.optimize(objective, n_trials=5000)
