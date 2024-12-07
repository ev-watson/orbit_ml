import plotly
import torch

import config

plotly.io.templates.default = 'plotly_dark'
torch.set_default_dtype(torch.float64) if not config.MAC else torch.set_default_dtype(torch.float32)

from .analysis import (
    separate_orbits,
    print_analysis,
    interp_test,
    gnn_test
)
from .torch_utils import (
    Scaler,
    GradientNormCallback,
    SEBlock,
    PredictorMixin,
)
from .coordinate_transformations import (
    sph_to_cart_windowed,
    cart_to_sph_windowed,
    cart_to_sph,
    sph_to_cart,
    mean_L_vector,
    alignment_matrix,
    apply_rotation,
    reflect_across_z,
    flat_plane
)
from .data_processing import (
    find_observation_times,
    convert_to_decimal_years,
    load_np
)
from .losses import (
    rmwe_loss,
    mape_loss,
    calc_mae,
    calc_mape,
    zero_one_approximation_loss
)
from .logging_utils import (
    print_err,
    print_block
)
from .misc import (
    clear_local_ckpt_files
)
from .numerical_methods import (
    socfdw,
    sixth_order_central_difference,
    get_movements
)
from .optuna_helpers import (
    sample_hyperparams,
    print_best_optuna,
    plot_pareto
)
