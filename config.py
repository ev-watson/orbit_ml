import inspect

SEED = 42
TYPE = 'gnn'
ROTATIONAL_EQUIVARIANCE = True

HIDDEN_DIM = 256
NUM_LAYERS = 3
LEARNING_RATE = 1e-5
USE_BN = False
USE_SE = False
REDUCTION = 2

MAX_EPOCHS = 20
ENABLE_EARLY_STOPPING = True
PATIENCE = 6
GRADIENT_CLIP_VAL = 1.2
WEIGHT_DECAY = 3e-5
DROP_RATE = 0.136
BATCH_SIZE = 64
SEQUENCE_LENGTH = 200  # EXPONENTIALLY AFFECTS TIME start low ~100

NUM_SAMPLES = 100000  # None for all available
STEP = 1
WINDOWED = True
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PIN_MEMORY = True
SCALE = True

LOG_ATTN = False
ON_STEP = False

GINPUTS_FILE = 'gnn_targets.npy'
INTERP_FILE = 'interp_data.npy'
SCALER_FILE = f'{TYPE}_scaler.pkl'  # None to turn off saving
STATE_FILE = 'model_state.pth'

MAC = False
if MAC:
    NUM_WORKERS = 0
    PREFETCH_FACTOR = None
    PIN_MEMORY = False

pre_registered_callers = {
    'data': 'ds',
    'model': 'model',
}

registry = {
    'interp': {
        'file': INTERP_FILE,
    },
    'gnn': {
        'file': GINPUTS_FILE,
    },
}


def register(type_key, module_key=None):
    """
    Decorator to register a class with a given key.
    :param type_key: str, which model type of key to register, from config.TYPE
    :param module_key: str, if module name is not pre-registered
    :return: decorated class
    """
    key = None
    if module_key:
        key = module_key
    else:
        caller_frame = inspect.stack()[1]
        caller_module = inspect.getmodule(caller_frame[0])

        if caller_module is None:
            raise RuntimeError("Cannot determine the caller module for registration.")

        module_name = caller_module.__name__.split('.')[-1]
        for k in pre_registered_callers.keys():
            if k in module_name:
                key = pre_registered_callers[k]
                break
        if key is None:
            raise RuntimeError("Cannot determine the caller module key for registration and 'module_key' argument was not set.")

    def decorator(cls):
        if type_key not in registry:
            raise KeyError(f"Type key {type_key} is not recognized (pre-registered).")
        registry[type_key][key] = cls
        return cls

    return decorator


def retrieve(key):
    """
    Retrieves the value of a given key in registry
    :param key: which key to grab within type_key
    :return: value of given key
    """
    return registry.get(TYPE).get(key)


hparams = {}


def update_hparams(new_hps):
    """
    Updates the hparams dictionary with new hparams values
    :param new_hps: dict
    :return: None
    """
    hparams.update(new_hps)
    return None
