# orbit_ml

Machine-learned orbital mechanics: predicting planetary trajectories with a message-passing graph neural network and distilling the learned force law into closed-form expressions via symbolic regression.

## Overview

The pipeline trains a Cranmer-style **message-passing graph neural network (MPNN)** on JPL Horizons ephemeris data, with the explicit goal of recovering the underlying force law from data alone. Each timestamp is a graph snapshot of the bodies in the system; a learned **edge function** `phi_e` produces a low-dimensional message between every pair of bodies, the messages are aggregated at each destination, and a learned **node function** `phi_v` maps the aggregated messages to the body's predicted acceleration. An L1 regularizer on the messages encourages a sparse bottleneck so that each active message channel can be pulled out and fit by **PySR** symbolic regression — recovering the form of Newton's law (and any small relativistic corrections) directly from real ephemerides.

The default system is Sun + Mercury (a 2-body graph). Adding more bodies is a one-liner extension via `extra_bodies` in `utils.data_processing.build_graph_snapshots`.

Key components:

- **MPNN with separate edge / node functions** (`models.MPNN`) — `phi_e: (V_src, V_dst) -> m`, `phi_v: (V, sum_j m_ji) -> a`. Optional SE-style channel gating inside each MLP block.
- **L1-regularized message bottleneck** — Cranmer's trick for symbolic distillation.
- **Per-snapshot graph dataset** (`data_construction.GraphDataset`) with custom collate; topology is fixed and shared across the batch.
- **Optuna hyperparameter optimization** — search space covers edge/node widths and depths, message dimension, L1 weight, optimizer/loss/activation.
- **Symbolic distillation of the edge function** (`pysr_main.py`) — finds analytic expressions for the most-active message channels and bundles the scaler so callers can compose back into physical units.
- **Trajectory integration** (`integration.py`) — propagates orbits using either Newton's law or the trained MPNN as the force model.

## Project Structure

```
config.py              # Hyperparameters, registry, file paths, MAC switch
models.py              # BaseArch + MPNN (graph network), SEMLP baseline, MLP/CNN/LSTM helpers
train.py               # Training entry point (Lightning)
eval.py                # Evaluation / inference driver, dispatches by config.TYPE
data_init.py           # Horizons ephemeris fetching
data_construction.py   # GraphDataset + GraphCollate-aware NNDataModule
hopt.py                # Optuna hyperparameter search (gnn / semlp / interp)
pysr_main.py           # PySR distillation of the trained MPNN's edge function
integration.py         # ODE orbit propagator (newton or trained-gnn force)
plot_advancement.py    # Perihelion-advancement diagnostic plots
orbit_anim.py          # Manim orbit animation
utils/                 # Losses, scalers, MLPBlock, scatter_sum, coords, analysis helpers
data/                  # Raw + processed data artifacts (gitignored except version_1.csv)
artifacts/             # Trained-model artifacts: scalers, optuna DBs, plots (gitignored)
notebooks/             # Jupyter workspaces (workspace.ipynb, symb_regr.ipynb)
scripts/               # SLURM / shell scripts for cluster submission
pysr/                  # Symbolic regression outputs (hall-of-fame CSVs, distilled.pkl)
hopt/                  # Optuna study checkpoints and logs
tlogs/                 # Lightning training logs and checkpoints (gitignored)
```

## Data layout

The MPNN is trained on a tensor of shape `(T, B, 10)` saved at `data/graph_data.npy` with column order

```
[mass, x, y, z, vx, vy, vz, ax, ay, az]
```

per body per snapshot. The first 7 columns are inputs (`MPNN.input_slice`), the last 3 are targets (`MPNN.output_slice`). Body 0 is the Sun (held fixed at origin in the heliocentric frame); body 1 is Mercury, with positions and velocities pulled from Horizons and accelerations obtained by sixth-order spline differentiation of the velocity timeseries (`utils.numerical_methods.get_movements`). The Sun's target row is held identically zero and excluded from the loss via a per-body `predict_mask`.

To build the file from a freshly fetched Horizons CSV:

```python
from utils import build_graph_snapshots
build_graph_snapshots('data/horizons.csv')          # default Sun + Mercury
# build_graph_snapshots('data/horizons.csv',
#     extra_bodies=[{'name': 'venus', 'mass': 4.8675e24, 'csv': 'data/venus.csv'}])
```

## Usage

All commands are run from the project root.

```bash
# Fetch Horizons ephemeris and write data/<name>.csv
bash scripts/run_data_init.sh

# Build the graph-snapshot tensor at data/graph_data.npy (one-off; do this before training)
python -c "from utils import build_graph_snapshots; build_graph_snapshots('data/horizons.csv')"

# Train the MPNN
bash scripts/run_train.sh

# Hyperparameter search (--model gnn | semlp | interp)
bash scripts/run_hopt.sh

# Symbolic distillation of the trained edge function
bash scripts/run_pysr.sh

# Evaluation / debug
bash scripts/run_eval.sh
```

## Model registry

`config.TYPE` selects which model + dataset pair the rest of the pipeline uses:

| `TYPE`   | Model class | Dataset class       | Description                                                               |
|----------|-------------|---------------------|---------------------------------------------------------------------------|
| `gnn`    | `MPNN`      | `GraphDataset`      | Cranmer-style message-passing GNN. Default and primary model.             |
| `semlp`  | `SEMLP`     | `SEMLPDataset`      | Legacy MLP-with-SE baseline (the original misnamed "GNN"). Flat features. |
| `interp` | `InterpMLP` | `InterpolationDataset` | Smaller MLP for the angular-momentum interpolation experiment.         |

## Requirements

- Python 3.9+
- PyTorch, PyTorch Lightning
- PySR, SymPy
- Optuna, SciPy, NumPy, joblib
- astroquery (for Horizons data)

## AI Disclosure

AI-assisted tools (Claude, Anthropic) were used during development of this repository.
