import os
import random

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import config
from models import GNN
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
equation_file = pysr_dir + 'hall_of_fame.csv'
evaluation_file = pysr_dir + 'evaluation.txt'

# --- SANITY CHECK ---
# targets = np.load(config.TARGETS_FILE)
# X = targets[slice(500000, 600000), 1].reshape(-1, 1)
# y = targets[slice(500000, 600000), 7].reshape(-1, 1)

# --- MAIN RUN ---
ckpt = 'hopt/checkpoints/gnn_nadam_328.ckpt'
gnnmodel = GNN.load_from_checkpoint(ckpt)

print_block("RETRIEVING DATA")
X, y = gnn_test(gnnmodel, ntrials=10240, batch_size=32, suppress=False, mape=True, SR=True)  # ntrials can be lowered to 10000

newton = (-1.32712e20 / X[:, 0] ** 2).reshape(-1, 1)  # GM value from NASA Solar Fact Sheet
with open(evaluation_file, 'w') as f:
    f.write("PROJECTED NEWTONIAN ACCURACY\n")
    f.write("MSE (PYSR DEFAULT)\n")
    f.write(f"{np.mean((y - newton) ** 2):.4g}\n")
    f.write("MAE\n")
    f.write(f"{np.mean(np.abs(y - newton)):.4g}\n")
    f.write("MAPE\n")
    f.write(f"{np.mean(np.abs((y - newton) / y)) * 100:.3g}%\n")

print_block("INITIATING REGRESSION")
split_index = int(0.8 * len(X))
x_train, x_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# set variable names based on input dimension, put cols into dict for easy grabbing later
variable_names = ['r'] if X.shape[1] == 1 else ['r', 'theta', 'phi', 'v_r', 'v_theta', 'v_phi']
column_dicts = dict(zip(variable_names, x_val.T))

# initialize PySRRegressor favoring complexity
model = PySRRegressor(
    niterations=25 if swift else 500,
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
    # select_k_features=4,
    tournament_selection_n=10,  # 10
    tournament_selection_p=0.86,  # 0.86
    should_optimize_constants=True,
    weight_optimize=0.002,  # 0.
    weight_simplify=0.004,  # 0.0020
    optimizer_nrestarts=3,  # 2
    optimize_probability=0.16,  # 0.14
    optimizer_iterations=10,  # 8
    parsimony=0.0028,  # 0.0032
    adaptive_parsimony_scaling=25.,  # 20.
    should_simplify=True,
    use_frequency=True,
    use_frequency_in_tournament=True,
    maxsize=15 if swift else 30,    # 20
    maxdepth=None,
    populations=3 * cores,
    population_size=33 if swift else 150,
    ncycles_per_iteration=200 if swift else 6500,    # 550
    procs=cores,
    random_state=seed,
    verbosity=1,
    tempdir=temp_dir,
    equation_file=equation_file,
    delete_tempfiles=True,
    turbo=True,
    warm_start=False,
    # cluster_manager='slurm' if not config.MAC else None,  # cluster manager is deprecated
)

# fit PySRRegressor on training data
model.fit(x_train, y_train, variable_names=variable_names)

# retrieve and simplify the best equation found
best_equation = sp.sympify(str(model.sympy()))
print("Best equation found:", best_equation)

# create a lambdified function from the best equation,
# sort symbols by order of appearance in variable_names for predictable function arguments
symbols_list = sorted(list(best_equation.free_symbols),
                      key=lambda s: variable_names.index(str(s)))
predict = sp.lambdify(symbols_list, best_equation, modules='numpy')
y_pred = predict(*[column_dicts[str(key)] for key in symbols_list])

# calculate evaluation metrics
mse = np.mean((y_val - y_pred) ** 2)
mae = np.mean(np.abs(y_pred - y_val))
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

mse_str = f"Val MSE: {mse:.6g}"
mae_str = f"Val MAE: {mae:.6g}"
mape_str = f"Val MAPE: {mape:.3f}%"
print(mse_str)
print(mae_str)
print(mape_str)

# append evaluation metrics to file
with open(evaluation_file, 'a') as f:
    f.write(f"Best equation found: {best_equation}\n")
    f.write(mse_str + "\n")
    f.write(mae_str + "\n")
    f.write(mape_str)

# get latex version of equation for plot label
latex_eq = sp.latex(best_equation)

# plot the results against newton, interactive
r = x_val[:, 0]
y_newton = (-1.32712e20 / r ** 2)
plt.plot(r, y_newton, label="Newton")
plt.scatter(r, y_val, s=10, label='Validation Data')
plt.scatter(r, y_pred, s=10, color='red', label='Model Prediction')
plt.text(0.05, 0.95, rf"Model: ${latex_eq}$", transform=plt.gca().transAxes, verticalalignment='top')
plt.legend()
plt.savefig("pysr_main.png")
plt.show()


"""
FOR USING WITH PREVIOUSLY KNOWN CONSTANTS
"""
# # Substitute the actual values of constants in the best equation
# def substitute_constants(equation, constants):
#     for symbol, value in constants.items():
#         equation = equation.subs(sp.Symbol(symbol), value)
#     return equation
#
#
# # Ensure the best equation is a SymPy expression
# best_equation_with_values = substitute_constants(best_equation, constants)
# print("Best equation with constant values:", best_equation_with_values)
