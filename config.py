import inspect

####################
#  BASIC SETTINGS  #
####################
SEED = 42
TYPE = 'gnn'
ROTATIONAL_EQUIVARIANCE = True
WINDOWED = False
MAC = True

####################
# MODEL PARAMETERS #
####################
HIDDEN_DIM = 256
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
USE_BN = False
USE_SE = True
SE_REDUCTION = 16

#####################
# TRAINING SETTINGS #
#####################
MAX_EPOCHS = 25
ENABLE_EARLY_STOPPING = True
PATIENCE = 7
GRADIENT_CLIP_VAL = 1.5
WEIGHT_DECAY = 7e-3
DROP_RATE = 0.5
DROPOUT_FREQUENCY = 3   # Every X layers a dropout layer will occur, so less is more frequent
SEQUENCE_LENGTH = 200  # EXPONENTIALLY AFFECTS TIME start low ~100

####################
#  BATCH AND DATA  #
####################
BATCH_SIZE = 64
NUM_SAMPLES = None  # None for all available
STEP = 1
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PIN_MEMORY = True
SCALE = True

####################
#  LOGGING/FLAGS   #
####################
LOG_ATTN = False
ON_STEP = False

####################
#    FILE NAMES    #
####################
GINPUTS_FILE = 'ginputs.npy'
TARGETS_FILE = 'gnn_targets.npy'
INTERP_FILE = 'interp_data.npy'
SCALER_FILE = f'{TYPE}_scaler.pkl'  # None to turn off saving
STATE_FILE = 'model_state.pth'

####################
#   MAC SETTINGS   #
####################
if MAC:
    NUM_SAMPLES = 100000
    NUM_WORKERS = 0
    PREFETCH_FACTOR = None
    PIN_MEMORY = False

####################
#     REGISTRY     #
####################
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
