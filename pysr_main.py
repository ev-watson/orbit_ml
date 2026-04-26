import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import config
from models import MPNN
from utils import print_block, gnn_test

if config.MAC:
    pass
from pysr import PySRRegressor

# for testing
swift = False  # if on, estimated time is ~30 seconds, extremely inaccurate

# set deterministic seed if desired, note regressor searches will not be deterministic even with a set seed
deterministic = False
seed = (config.SEED if deterministic else random.randrange(2**32 - 1))
print_block(f"SEED: {seed}")

# set number of cores based on environment
cores = 12 if config.MAC else 128

pysr_dir = 'pysr/'
temp_dir = 'pysr_temp'
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(pysr_dir, exist_ok=True)
evaluation_file = pysr_dir + 'evaluation.txt'

# Cranmer recipe -- distill the edge function phi_e of the trained message-passing GNN.
# The edge function takes (V_src, V_dst) and produces a low-dim message m_ij; under L1
# regularization a small subset of message channels carry signal. We fit a closed-form
# expression for each active channel as a function of the source and destination node
# features. Once those expressions are known, the full force law is the per-edge sum.

ckpt = 'tlogs/checkpoints/mpnn_final.ckpt'
gnnmodel = MPNN.load_from_checkpoint(ckpt)

print_block("RETRIEVING EDGE-FUNCTION ARTIFACTS")
edge_inputs, messages, accel_target, accel_pred = gnn_test(
    gnnmodel, ntrials=10240, batch_size=32, suppress=False, mape=True, SR=True
)
# edge_inputs: [N, M, 2 * F_in]; messages: [N, M, msg_dim]
# Sun+Mercury default has M = 2 directed edges; index 0 is the Sun -> Mercury edge.
sun_to_merc = 0
X = edge_inputs[:, sun_to_merc, :]                # [N, 2 * F_in]
M = messages[:, sun_to_merc, :]                   # [N, msg_dim]

print_block("MEASURING MESSAGE-CHANNEL ACTIVITY (Cranmer)")
channel_std = M.std(axis=0)                       # [msg_dim]
top_k = min(int(np.argsort(-channel_std).size), 4)
active = np.argsort(-channel_std)[:top_k]         # indices of the most active channels
print(f"Top {top_k} active message channels (by std): {active.tolist()}")
print(f"Channel std (first 10 shown): {np.round(channel_std[:10], 6)}")

# Newton baseline computed on the *unscaled* destination Mercury position. Because phi_e was
# trained on scaled inputs, the symbolic regression below operates in the same scaled space.
# The scaler bundle is shipped alongside the equations so the closed-form expression can be
# composed back into physical units downstream.
scalers = joblib.load(config.SCALER_FILE) if config.SCALE else None
F_in = X.shape[-1] // 2
src_var_names = ['mass_src', 'x_src', 'y_src', 'z_src', 'vx_src', 'vy_src', 'vz_src'][:F_in]
dst_var_names = ['mass_dst', 'x_dst', 'y_dst', 'z_dst', 'vx_dst', 'vy_dst', 'vz_dst'][:F_in]
variable_names = src_var_names + dst_var_names

split_index = int(0.8 * len(X))
x_train, x_val = X[:split_index], X[split_index:]

# Build a single PySRRegressor and reuse it across active channels (separate fits, separate
# equation files). Operator catalog matches the legacy pipeline so prior tuning still applies.
def build_regressor(equation_file):
    return PySRRegressor(
        niterations=25 if swift else 100,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["square", "cube", "sqrt", "sin", "cos", "tan", "exp"],
        constraints={
            "^": (-1, 3),
            "sqrt": 2,
            "square": 2,
            "cube": 2,
            "sin": 3,
            "cos": 3,
            "tan": 3,
            "exp": 3,
        },
        nested_constraints={
            "sin": {"sin": 0, "cos": 0, "tan": 0, "exp": 0},
            "cos": {"cos": 0, "sin": 0, "tan": 0, "exp": 0},
            "tan": {"tan": 0, "sin": 0, "cos": 0, "exp": 0},
            "square": {"square": 0, "cube": 0, "sqrt": 0, "exp": 0},
            "cube": {"square": 0, "cube": 0, "sqrt": 0, "exp": 0},
            "sqrt": {"square": 0, "cube": 0, "sqrt": 0, "exp": 0},
            "exp": {"exp": 0, "square": 0, "cube": 0, "sqrt": 0, "sin": 0, "cos": 0, "tan": 0},
        },
        model_selection="best",
        precision=32 if swift else 64,
        batching=True,
        batch_size=128 if swift else 32,
        tournament_selection_n=5,
        tournament_selection_p=0.75,
        should_optimize_constants=True,
        weight_optimize=0.,
        weight_simplify=0.0004,
        optimizer_nrestarts=2,
        optimize_probability=0.12,
        optimizer_iterations=8,
        parsimony=0.0010,
        adaptive_parsimony_scaling=30.,
        should_simplify=True,
        use_frequency=True,
        use_frequency_in_tournament=True,
        maxsize=15 if swift else 30,
        maxdepth=None,
        populations=3 * cores,
        population_size=33 if swift else 150,
        ncycles_per_iteration=200 if swift else 5000,
        procs=cores,
        random_state=seed,
        verbosity=1,
        tempdir=temp_dir,
        equation_file=equation_file,
        delete_tempfiles=True,
        turbo=True,
        warm_start=True,
    )


# --- DISTILL EACH ACTIVE MESSAGE CHANNEL ---
fitted_expressions = {}
with open(evaluation_file, 'w') as f:
    f.write(f"MPNN edge-function distillation\n")
    f.write(f"Top-{top_k} active channels (by std): {active.tolist()}\n")
    f.write(f"Channel stds (full): {channel_std.tolist()}\n")
    f.write("\n")

for k in active:
    print_block(f"INITIATING REGRESSION on message channel {int(k)}")
    eq_file = f"{pysr_dir}hall_of_fame_edge_ch{int(k)}.csv"
    regressor = build_regressor(eq_file)
    y_train, y_val = M[:split_index, k], M[split_index:, k]

    regressor.fit(x_train, y_train, variable_names=variable_names)
    best_equation = sp.sympify(str(regressor.sympy()))
    print(f"Channel {int(k)} best equation: {best_equation}")

    symbols_list = sorted(list(best_equation.free_symbols),
                          key=lambda s: variable_names.index(str(s)))
    predict = sp.lambdify(symbols_list, best_equation, modules='numpy')
    column_dicts = dict(zip(variable_names, x_val.T))
    y_pred = np.array(predict(*[column_dicts[str(key)] for key in symbols_list])).ravel()
    y_val = np.array(y_val).ravel()

    mse = np.mean((y_val - y_pred) ** 2)
    mae = np.mean(np.abs(y_pred - y_val))
    denom = np.where(np.abs(y_val) > 1e-12, y_val, 1.0)
    mape = np.mean(np.abs((y_val - y_pred) / denom)) * 100
    fitted_expressions[int(k)] = best_equation

    with open(evaluation_file, 'a') as f:
        f.write(f"--- Channel {int(k)} ---\n")
        f.write(f"Equation: {best_equation}\n")
        f.write(f"Val MSE: {mse:.6g}\n")
        f.write(f"Val MAE: {mae:.6g}\n")
        f.write(f"Val MAPE: {mape:.3f}%\n\n")

    # Per-channel diagnostic plot: x-axis is dst Mercury radius (a natural physical proxy).
    if 'x_dst' in column_dicts and 'y_dst' in column_dicts and 'z_dst' in column_dicts:
        r_dst = np.sqrt(column_dicts['x_dst'] ** 2
                        + column_dicts['y_dst'] ** 2
                        + column_dicts['z_dst'] ** 2)
        order = np.argsort(r_dst)
        plt.figure()
        plt.scatter(r_dst[order], y_val[order], s=8, label='Learned message')
        plt.scatter(r_dst[order], y_pred[order], s=8, color='red', label='SR fit')
        plt.text(0.05, 0.95, rf"$m^{{({int(k)})}} = {sp.latex(best_equation)}$",
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.xlabel("|r_dst|  (scaled)")
        plt.ylabel(f"message channel {int(k)}")
        plt.legend()
        plt.savefig(f"{pysr_dir}edge_ch{int(k)}.png")
        plt.close()


# --- BUNDLE EVERYTHING THE DOWNSTREAM INTEGRATOR NEEDS ---
distilled = {
    'edge_expressions': fitted_expressions,        # {channel_index: sympy expr}
    'variable_names': variable_names,
    'active_channels': [int(k) for k in active],
    'channel_std': channel_std,
    'edge_index_used': sun_to_merc,
    'scalers': scalers,                            # so callers can compose into physical units
}
joblib.dump(distilled, f"{pysr_dir}distilled.pkl")
print_block(f"Saved distilled edge function to {pysr_dir}distilled.pkl")
