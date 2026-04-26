import numpy as np
import torch
import joblib

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
    mae = calc_mae(g, t, axis=axis)
    if not suppress:
        print_block(f"RANDOM INPUT TESTING TRIALS: {ntrials}", err=err)
        print_block(f"MAE: {mae:.6g}", err=err)
        if verbose:
            print_block("PREDICTIONS:", err=err)
            print(g)
            print_block("TARGETS:", err=err)
            print(t)

    if mape:
        mape_val = calc_mape(g, t, axis=axis)
        if not suppress:
            print_block(f"MAPE: {mape_val:.6g}%", err=err)
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
    sphinputs = np.load('data/sphinputs.npy')
    r_data = sphinputs[:, 1]
    phi_data = sphinputs[:, 3]

    idx = np.random.randint(len(sphinputs) - 1, size=ntrials)
    sampled_r = r_data[idx]
    sampled_phi = phi_data[idx]
    vp_target = sphinputs[idx, -1]

    pred_vals = model.predict(np.vstack((sampled_r, sampled_phi)).T)

    return print_analysis(pred_vals, vp_target, ntrials, mape, suppress, err, verbose, axis=mean_axis)


def semlp_test(model, ntrials=100, batch_size=1024, mape=False, suppress=False, err=False, verbose=False,
               mean_axis=None, SR=False):
    """
    Random-input testing for the legacy SEMLP baseline (flat per-timestep features). Pulls samples
    from ``config.TARGETS_FILE`` and routes inputs through ``model.predict``.
    :param model: SEMLP model
    :param ntrials: int, number of trials
    :param batch_size: int, batch size for prediction, default 1024
    :param mape: bool, enable mape
    :param suppress: bool, if true, suppresses print statements.
    :param err: bool, enable printing to stderr as well.
    :param verbose: bool, enable verbose output
    :param mean_axis: int, axis along which to calculate analysis statistics, None for entire array
    :param SR: bool, if true, returns (inputs, predictions) instead of accuracy metrics
    :return: float or tuple of floats / arrays depending on flags
    """
    if not suppress:
        print_block("BEGINNING RANDOM SEMLP TESTING", err=err)

    targets = np.load(config.TARGETS_FILE)
    inp_slice = config.retrieve('model').input_slice
    targ_slice = config.retrieve('model').output_slice
    ntargs = config.retrieve('model').output_dim
    if SR:
        idx = np.random.randint(len(targets) - ntrials - 1)
        sampled = targets[idx:idx + ntrials]
        pred_vals = torch.empty((ntrials, ntargs))
    elif config.WINDOWED:
        s = config.SEQUENCE_LENGTH
        idx = np.random.randint(len(targets) - s + 1, size=ntrials)
        sampled = np.array([targets[i:i + s] for i in idx])
        pred_vals = torch.empty((ntrials, s, ntargs))
    else:
        idx = np.random.randint(len(targets) - 1, size=ntrials)
        sampled = np.array([targets[i] for i in idx])
        pred_vals = torch.empty((ntrials, ntargs))

    device = next(model.parameters()).device
    input_data = torch.from_numpy(sampled[..., inp_slice]).to(device=device, dtype=torch.get_default_dtype())
    a_target = torch.from_numpy(sampled[..., targ_slice]).to(device=device, dtype=torch.get_default_dtype())
    pred_vals = pred_vals.to(device)

    num_batches = int(np.ceil(ntrials / batch_size))
    if not suppress:
        print_block(f"{ntrials} TRIALS, {num_batches} BATCHES of {batch_size} SIZE", err=err)
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, ntrials)
        batch_input = input_data[start_idx:end_idx]
        pred_vals[start_idx:end_idx] = model.predict(batch_input)

    if SR:
        if not suppress:
            print_analysis(pred_vals, a_target, ntrials, mape, suppress, err, verbose, axis=mean_axis)
        return input_data.cpu().numpy(), pred_vals.cpu().numpy()

    if not suppress:
        print_block("BEGINNING ANALYSIS", err=err)
    return print_analysis(pred_vals, a_target, ntrials, mape, suppress, err, verbose, axis=mean_axis)


def gnn_test(model, ntrials=100, batch_size=1024, mape=False, suppress=False, err=False, verbose=False,
             mean_axis=None, SR=False):
    """
    Random-input testing for the message-passing GNN. Pulls random graph snapshots from the
    active graph data tensor from the config registry, runs them through the model, and reports
    per-component MAE on the predict-masked nodes (Mercury and any extra bodies).

    When ``SR`` is true, returns the raw artifacts needed to symbolically distill the learned
    edge function: the (V_src, V_dst) edge inputs and the messages produced by ``phi_e``.

    :param model: MPNN instance.
    :param ntrials: int, number of random snapshots to sample.
    :param batch_size: int, mini-batch size used while iterating through the snapshots.
    :param mape: bool, enable MAPE alongside MAE.
    :param suppress: bool, if true silence reporting prints.
    :param err: bool, also print to stderr.
    :param verbose: bool, dump predictions and targets when reporting.
    :param mean_axis: int or None, axis along which to reduce when computing MAE/MAPE.
    :param SR: bool, if true return ``(edge_input, messages, accel_target, accel_pred)`` for SR.
    :return: depends on ``SR``: either accuracy metrics or the SR artifacts above.
    """
    if not suppress:
        print_block("BEGINNING RANDOM GRAPH GNN TESTING", err=err)

    snapshots = np.load(config.retrieve('file'))            # [T, B, F]
    T, num_bodies, _ = snapshots.shape
    inp_slice = config.retrieve('model').input_slice
    targ_slice = config.retrieve('model').output_slice
    ntargs = config.retrieve('model').output_dim

    idx = np.random.randint(T, size=ntrials)
    sampled = snapshots[idx]                                # [ntrials, B, F]

    device = next(model.parameters()).device
    V = torch.from_numpy(sampled[..., inp_slice]).to(device=device, dtype=torch.get_default_dtype())
    Y = torch.from_numpy(sampled[..., targ_slice]).to(device=device, dtype=torch.get_default_dtype())
    scalers = joblib.load(config.SCALER_FILE) if config.SCALE else None
    V_model = scalers['input_scaler'].transform(V) if scalers is not None else V

    src_index, dst_index = _fully_connected_edges(num_bodies)
    src_index = src_index.to(device)
    dst_index = dst_index.to(device)
    predict_mask = torch.arange(1, num_bodies, dtype=torch.long, device=device)

    preds = torch.empty((ntrials, num_bodies, ntargs), device=device)
    edge_inputs = []
    messages = []

    num_batches = int(np.ceil(ntrials / batch_size))
    if not suppress:
        print_block(f"{ntrials} TRIALS, {num_batches} BATCHES of {batch_size} SIZE", err=err)

    model.eval()
    with torch.no_grad():
        for b in range(num_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, ntrials)
            batch = {
                'nodes': V[s:e],
                'src_index': src_index,
                'dst_index': dst_index,
                'predict_mask': predict_mask,
            }
            preds[s:e] = model.predict(batch)
            if SR:
                # Distillation targets are the internal edge-function inputs/messages, so this
                # intentionally uses the same scaled node space seen by phi_e during training.
                scaled_batch = dict(batch)
                scaled_batch['nodes'] = V_model[s:e]
                ei, m = model.edge_messages(scaled_batch)
                edge_inputs.append(ei)
                messages.append(m)

    g = preds[:, predict_mask, :]
    t = Y[:, predict_mask, :]

    if SR:
        edge_inputs = np.concatenate(edge_inputs, axis=0)   # [ntrials, M, 2*F_in]
        messages = np.concatenate(messages, axis=0)         # [ntrials, M, msg_dim]
        if not suppress:
            print_analysis(g, t, ntrials, mape, suppress, err, verbose, axis=mean_axis)
        return edge_inputs, messages, t.cpu().numpy(), g.cpu().numpy()

    if not suppress:
        print_block("BEGINNING ANALYSIS", err=err)
    return print_analysis(g, t, ntrials, mape, suppress, err, verbose, axis=mean_axis)


def _fully_connected_edges(num_bodies):
    """
    Build directed edge_index tensors for a fully connected graph of ``num_bodies`` nodes
    (no self-loops). Mirrors the helper in NNDataModule so ``gnn_test`` can run standalone.

    :param num_bodies: int.
    :return: tuple (src_index, dst_index) of torch.LongTensor with shape (M,).
    """
    src, dst = [], []
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                src.append(i)
                dst.append(j)
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long))
