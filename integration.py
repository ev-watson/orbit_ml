import argparse

import joblib
import numpy as np
import torch
from scipy.integrate import solve_ivp

import config
from models import MPNN
from utils import print_block


def newton_force(r_vec, gm=config.GM_SUN):
    """
    Newtonian gravitational acceleration of a test body at displacement ``r_vec`` from the Sun.

    :param r_vec: array-like of shape (3,), heliocentric position vector in meters.
    :param gm: float, GM_sun in m^3 / s^2.
    :return: np.ndarray of shape (3,), acceleration in m / s^2.
    """
    r = np.linalg.norm(r_vec)
    return -gm * r_vec / r ** 3


def gnn_force(model, r_vec, v_vec,
              merc_mass=config.M_MERC,
              sun_mass=config.M_SUN,
              scalers=None,
              num_bodies=2):
    """
    Acceleration on Mercury predicted by the trained message-passing GNN. Builds a single-snapshot
    Sun + Mercury graph, scales it the way the model was trained, runs a forward pass, and pulls
    Mercury's predicted acceleration row.

    :param model: MPNN, the trained graph network in eval mode.
    :param r_vec: array-like of shape (3,), Mercury heliocentric position (m).
    :param v_vec: array-like of shape (3,), Mercury heliocentric velocity (m/s).
    :param merc_mass: float, Mercury mass (kg).
    :param sun_mass: float, Sun mass (kg).
    :param scalers: dict, scaler bundle saved during training. None -> use raw values.
    :param num_bodies: int, total bodies in the snapshot. Must match training topology.
    :return: np.ndarray of shape (3,), predicted acceleration in m / s^2.
    """
    snap = np.zeros((1, num_bodies, 7), dtype=np.float64)
    snap[0, 0, 0] = sun_mass
    snap[0, 1, 0] = merc_mass
    snap[0, 1, 1:4] = r_vec
    snap[0, 1, 4:7] = v_vec

    device = next(model.parameters()).device
    if scalers is not None:
        snap_t = torch.from_numpy(snap).to(device=device, dtype=torch.get_default_dtype())
        snap_t = scalers['input_scaler'].transform(snap_t)
    else:
        snap_t = torch.from_numpy(snap).to(device=device, dtype=torch.get_default_dtype())

    src, dst = [], []
    for i in range(num_bodies):
        for j in range(num_bodies):
            if i != j:
                src.append(i)
                dst.append(j)
    batch = {
        'nodes': snap_t,
        'src_index': torch.tensor(src, dtype=torch.long, device=device),
        'dst_index': torch.tensor(dst, dtype=torch.long, device=device),
    }

    with torch.no_grad():
        pred_scaled = model(batch)                 # [1, B, 3] in scaled-acceleration space
    if scalers is not None:
        pred = scalers['target_scaler'].inverse_transform(pred_scaled)
    else:
        pred = pred_scaled
    return pred[0, 1].cpu().numpy()                # Mercury row


def ode_rhs(t, y, force_fn):
    """
    Six-state RHS for a heliocentric Newtonian/GR-corrected orbit integration.
    State vector ``y = [x, y, z, vx, vy, vz]``; the RHS applies the supplied ``force_fn``
    (Newton or learned) at the current point.

    :param t: float, time (unused; included for solve_ivp compatibility).
    :param y: array-like of shape (6,), state vector.
    :param force_fn: callable, takes (r_vec, v_vec) and returns acceleration of shape (3,).
    :return: np.ndarray of shape (6,), dy/dt.
    """
    r = y[:3]
    v = y[3:]
    a = force_fn(r, v)
    return np.concatenate([v, a])


def initial_state(merc_csv=None):
    """
    Pull Mercury's first heliocentric (x, y, z, vx, vy, vz) from the Horizons CSV used to build
    the graph dataset. Falls back to a hard-coded perihelion-near initial condition if no CSV
    path is supplied.

    :param merc_csv: str, optional path to a Mercury Horizons CSV in
        [JDTDB, X, Y, Z, VX, VY, VZ] form (km, km/s).
    :return: np.ndarray of shape (6,) in meters / m/s.
    """
    if merc_csv is None:
        # Mercury near aphelion in 1870, in heliocentric ICRF (m, m/s). Approximate placeholder.
        return np.array([6.98e10, 0.0, 0.0, 0.0, 3.886e4, 0.0])
    import pandas as pd
    df = pd.read_csv(merc_csv)
    row = df.iloc[0]
    return np.array([row['X'], row['Y'], row['Z'], row['VX'], row['VY'], row['VZ']]) * 1000.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbit propagator using Newton or learned force law")
    parser.add_argument('--mode', choices=['newton', 'gnn'], default='newton',
                        help='Force law: analytic Newton or trained MPNN.')
    parser.add_argument('--ckpt', type=str, default='tlogs/checkpoints/mpnn_final.ckpt',
                        help='Checkpoint path for --mode gnn')
    parser.add_argument('--scaler', type=str, default=config.SCALER_FILE,
                        help='Scaler bundle path used during training (None to skip)')
    parser.add_argument('--merc-csv', type=str, default=None,
                        help='Mercury Horizons CSV; if given, supplies the initial condition')
    parser.add_argument('--t-end', type=float, default=88 * 86400.0,
                        help='Integration end time in seconds (default ~1 Mercury year)')
    parser.add_argument('--n-out', type=int, default=20000,
                        help='Number of output samples')
    args = parser.parse_args()

    if args.mode == 'newton':
        force_fn = lambda r, v: newton_force(r)
    else:
        model = MPNN.load_from_checkpoint(args.ckpt)
        model.eval()
        scalers = joblib.load(args.scaler) if args.scaler else None
        force_fn = lambda r, v: gnn_force(model, r, v, scalers=scalers)

    y0 = initial_state(args.merc_csv)
    t_span = (0.0, args.t_end)
    t_eval = np.linspace(*t_span, args.n_out)
    print_block(f"Integrating {args.mode} force law from t=0 to t={args.t_end:.3g} s")

    sol = solve_ivp(
        ode_rhs, t_span, y0, args=(force_fn,),
        method='Radau', t_eval=t_eval, dense_output=True,
        rtol=1e-9, atol=1e-3,
    )

    print(f"Solver status: {sol.status}")
    print(f"Solver message: {sol.message}")
    out = np.column_stack([sol.t, sol.y.T])
    np.save(f"{config.ARTIFACTS_DIR}/integrated_{args.mode}.npy", out)
    print_block(f"Saved trajectory to {config.ARTIFACTS_DIR}/integrated_{args.mode}.npy")
