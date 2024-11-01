import numpy as np
import torch

import config
from utils.logging_utils import print_block
from utils.losses import calc_mae, calc_mape


def separate_orbits(inputs):
    """
    Finds orbit indices based on wrapping from +pi to -pi
    Then splits original dataset into orbits and finds avg theta of all orbits
    and returns the 2 orbits with biggest difference in theta
    :param inputs: array-like of shape (~, 3), cartesian position components
    :return: a tuple of 2 arrays each of shape (~, 3), lowest and highest theta orbits respectively
    """
    phi = np.arctan2(inputs[:, 1], inputs[:, 0])

    wrap_indices = np.where(np.diff(np.signbit(phi)) & (np.abs(phi[:-1]) > 3.1))[0]

    orbits = []
    start_idx = wrap_indices[0]
    for wrap_idx in wrap_indices[1:]:
        orbits.append(inputs[start_idx:wrap_idx + 1])
        start_idx = wrap_idx + 1

    avg_theta = [np.mean(np.arccos(orbit[:, 2] / np.linalg.norm(orbit, axis=1))) for orbit in orbits]

    min_idx, max_idx = np.argmin(avg_theta), np.argmax(avg_theta)

    orbit1_points = orbits[min_idx]
    orbit2_points = orbits[max_idx]

    return orbit1_points, orbit2_points


def print_analysis(g, t, ntrials, mape, suppress, err, verbose, axis=None):
    """
    Helper function for random tests
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    guess = torch.from_numpy(g).float().to(device)
    target = torch.from_numpy(t).float().to(device)
    mae = calc_mae(guess, target, axis=axis)
    if not suppress:
        print_block(f"RANDOM INPUT TESTING TRIALS: {ntrials}", err=err)
        print_block(f"MAE: {mae}", err=err)
        if verbose:
            print_block("PREDICTIONS:", err=err)
            print(guess)
            print_block("TARGETS:", err=err)
            print(target)

    if mape:
        mape_val = calc_mape(guess, target, axis=axis)
        if not suppress:
            print_block(f"MAPE: {mape_val}%", err=err)
        return mae, mape_val
    else:
        return mae


def interp_test(model, ntrials=100, mape=False, suppress=False, err=False, verbose=False, mean_axis=None):
    """
    Performs random input testing to evaluate the accuracy of Mercury's orbit interpolation.
    :param model: InterpMLP model
    :param ntrials: int, number of trials
    :param mape: bool, enable mape
    :param suppress: bool, if true, suppresses print statements.
    :param err: bool, enable printing to stderr as well.
    :param verbose: bool, enable verbose output
    :param mean_axis: int, axis along which to calculate analysis statistics, None for entire array
    :return: float or tuple of floats, mae or (mae, mape) if mape.
    """
    sphinputs = np.load('sphinputs.npy')
    r_data = sphinputs[:, 1]
    phi_data = sphinputs[:, 3]

    idx = np.random.randint(len(sphinputs) - 1, size=ntrials)
    sampled_r = r_data[idx]
    sampled_phi = phi_data[idx]
    vp_target = sphinputs[idx, -1]

    pred_vals = model.predict(np.vstack((sampled_r, sampled_phi)).T)

    return print_analysis(pred_vals, vp_target, ntrials, mape, suppress, err, verbose, axis=mean_axis)


def gnn_test(model, ntrials=100, mape=False, suppress=False, err=False, verbose=False, mean_axis=None, SR=False):
    """
    Performs random input testing to evaluate the accuracy of GNN's acceleration prediction.
    :param model: GNN model
    :param ntrials: int, number of trials
    :param mape: bool, enable mape
    :param suppress: bool, if true, suppresses print statements.
    :param err: bool, enable printing to stderr as well.
    :param verbose: bool, enable verbose output
    :param mean_axis: int, axis along which to calculate analysis statistics, None for entire array
    :param SR: bool, if true, returns inputs/outputs to be used in symbolic regression
    :return: float or tuple of floats, mae or (mae, mape) if mape.
    """
    if not suppress:
        print_block("BEGINNING RANDOM GNN TESTING", err=err)

    targets = np.load('gnn_targets.npy')
    inp_slice = config.retrieve('model').input_slice
    targ_slice = config.retrieve('model').output_slice
    ntargs = config.retrieve('model').output_dim
    if config.WINDOWED:
        s = config.SEQUENCE_LENGTH
        idx = np.random.randint(len(targets) - s + 1, size=ntrials)
        sampled = np.array([targets[i:i + s] for i in idx])  # shape [ntrials, s, f]
        pred_vals = np.empty((ntrials, s, ntargs))
    else:
        idx = np.random.randint(len(targets) - 1, size=ntrials)
        sampled = np.array([targets[i] for i in idx])
        pred_vals = np.empty((ntrials, ntargs))

    input_data = sampled[..., inp_slice]
    a_target = sampled[..., targ_slice]

    # Finds closest power of 2 that will make batch_size and num_batches as even as possible
    batch_size = config.BATCH_SIZE if config.BATCH_SIZE and config.BATCH_SIZE < ntrials else 2 ** round(np.log2(np.sqrt(ntrials)))
    num_batches = int(np.ceil(ntrials / batch_size))
    if not suppress:
        print_block(f"{ntrials} TRIALS, {num_batches} BATCHES of {batch_size} SIZE", err=err)
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, ntrials)
        batch_input = input_data[start_idx:end_idx]
        batch_pred = model.predict(batch_input)
        pred_vals[start_idx:end_idx] = batch_pred.cpu()

    if SR:
        return input_data, pred_vals

    if not suppress:
        print_block("BEGINNING ANALYSIS", err=err)
    return print_analysis(pred_vals, a_target, ntrials, mape, suppress, err, verbose, axis=mean_axis)
