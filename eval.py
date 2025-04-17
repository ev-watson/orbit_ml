import argparse

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger

from data_construction import *

seed = config.SEED if config.SEED else np.random.randint(1, 10000)
seed_everything(seed)
print_block(f"SEED: {seed}", err=True)

torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description="testing")
parser.add_argument("--ntrials", "-n", type=int, default=100,
                    help="Number of random trials to conduct, default 100")
parser.add_argument("--batch-size", "-b", type=int, default=1024,
                    help="Batch size for predictions, default 1024")
parser.add_argument('--ckpt', '-c', type=str, default=None,
                    help='Name of checkpoint file, if provided, will load from checkpoint rather than state dict')
parser.add_argument('--skip-stage', '-s', type=str, default='none',
                    help="What stage to skip, can be one of 'none' (default), 'data' for data module testing, "
                         "or 'random' for random testing")
args = parser.parse_args()

if args.ckpt:
    model = config.retrieve('model').load_from_checkpoint(args.ckpt)
    print_block("MODEL LOADED FROM CHECKPOINT")
else:
    model = config.retrieve('model')()
    model.load_state_dict(torch.load(config.STATE_FILE, weights_only=True))
    print_block("MODEL LOADED FROM STATE FILE")

data_module = NNDataModule()
if args.skip_stage != 'data':
    print_block("DATAMODULE TESTING")

    trainer = Trainer(accelerator='gpu',
                      devices=-1,
                      benchmark=True,
                      logger=TensorBoardLogger('testing', 'test')
                      )

    trainer.test(model=model, datamodule=data_module)

    print_block(f"{trainer.callback_metrics['test_loss'].item()}")
else:
    print_block("SKIPPING DATAMODULE TESTING")

if args.skip_stage != 'random':

    if config.TYPE == 'interp':
        interp_test(model, ntrials=args.ntrials, mape=True)
    else:
        gnn_test(model, ntrials=args.ntrials, batch_size=args.batch_size, mape=True, verbose=False, mean_axis=None)
else:
    print_block("SKIPPING RANDOM INPUT TESTING")
