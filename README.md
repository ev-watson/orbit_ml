# orbit_ml

Machine-learned orbital mechanics: predicting planetary trajectories with graph neural networks and symbolic regression.

## Overview

orbit_ml trains a **rotationally-equivariant Graph Neural Network (GNN)** on JPL Horizons ephemeris data to learn orbital dynamics, then distills the learned representation into closed-form expressions via **PySR symbolic regression**. The pipeline currently targets Mercury's orbit as a test case for recovering relativistic precession corrections from data alone.

Key components:
- **GNN with SE blocks**: Squeeze-and-Excitation channel attention over a temporal graph of positional/velocity states, with optional rotational equivariance enforced at the architecture level
- **Optuna hyperparameter optimization**: multi-study sweeps (NAdam, AdamW) with importance analysis
- **PySR symbolic distillation**: extracts interpretable analytic expressions from the trained GNN's predictions
- **Numerical integration**: ODE-based orbit propagation using the learned angular momentum field

## Project Structure

```
config.py              # Hyperparameters, flags, environment settings
models.py              # GNN architecture (BaseArch + equivariant layers)
train.py               # Training entry point (Lightning)
data_construction.py   # Horizons data loading, graph construction, scaling
hopt.py                # Optuna hyperparameter search
pysr_main.py           # PySR symbolic regression driver
integration.py         # ODE orbit integration with learned force model
utils/                 # Losses, logging, analysis, coordinate transforms, numerical methods
pysr/                  # Symbolic regression outputs (hall of fame, plots)
hopt/                  # Hyperparameter study checkpoints and logs
job_scripts/           # SLURM / shell scripts for cluster submission
```

## Usage

```bash
# Prepare data from Horizons ephemeris
bash run_data_init.sh

# Train GNN
bash run_train.sh

# Hyperparameter search
bash run_hopt.sh

# Symbolic regression
bash run_pysr.sh

# Evaluate / debug
bash run_eval.sh
```

## Requirements

- Python 3.9+
- PyTorch, PyTorch Lightning
- PySR, SymPy
- Optuna, SciPy, NumPy
- astroquery (for Horizons data)
