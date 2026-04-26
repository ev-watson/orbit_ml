import os

import dask.dataframe as dd
import numpy as np
from astropy.time import Time
from tqdm import tqdm

import config
from utils.numerical_methods import get_movements


def find_observation_times(data):
    """
    Finds realistic observation times of Mercury based on elongation angle and sun elevation as viewed from Earth
    :param data: array of shape (~, 7), column vectors need to be in order: [t, mx, my, mz, ex, ey, ez]
    :return: list of valid times for an observation
    """
    t = data[:, 0]
    mx, my, mz = data[:, 1], data[:, 2], data[:, 3]
    ex, ey, ez = data[:, 4], data[:, 5], data[:, 6]

    rel_x = mx - ex
    rel_y = my - ey
    rel_z = mz - ez

    r_me = np.vstack([rel_x, rel_y, rel_z]).T
    r_es = np.vstack([-ex, -ey, -ez]).T

    dot_product = np.einsum('ij,ij->i', r_me, r_es)
    norm_r_me = np.linalg.norm(r_me, axis=1)
    norm_r_es = np.linalg.norm(r_es, axis=1)

    cos_angle = dot_product / (norm_r_me * norm_r_es)
    angle = np.arccos(np.clip(cos_angle, -1, 1))

    elongation_threshold = np.radians(15)

    solar_elevation = np.arcsin(-ez / norm_r_es)

    valid_indices = (angle > elongation_threshold) & (solar_elevation < 0.1)

    return t[valid_indices]


def convert_to_decimal_years(jdtbd_array, chunk_size=100000):
    """
    Convert an array of Julian dates to decimal years in chunks.

    :param jdtbd_array: numpy array of Julian dates
    :param chunk_size: size of each chunk for processing
    :return: numpy array of decimal years
    """
    decimal_years = []
    progress_bar = tqdm(total=len(jdtbd_array), desc="Converting Julian dates to decimal years")

    for start in range(0, len(jdtbd_array), chunk_size):
        end = min(start + chunk_size, len(jdtbd_array))
        chunk = jdtbd_array[start:end]
        date_time = Time(chunk, format='jd', scale='tdb')
        decimal_years.append(date_time.decimalyear)
        progress_bar.update(len(chunk))

    progress_bar.close()
    return np.concatenate(decimal_years)


def build_graph_snapshots(merc_csv,
                          out_file=None,
                          extra_bodies=None,
                          merc_mass=config.M_MERC,
                          sun_mass=config.M_SUN,
                          chunk_size=400000,
                          nrows=None,
                          step=1):
    """
    Build per-timestep multi-body graph snapshots in the heliocentric frame for the message-passing
    GNN. The Sun is body index 0 (held fixed at origin) and Mercury is body index 1; additional
    bodies follow in the order given by ``extra_bodies``.

    Each snapshot has shape (B, F) with column order
    ``[mass, x, y, z, vx, vy, vz, ax, ay, az]``. The first 7 columns are model inputs, the last 3
    are targets. The Sun's target row is held identically zero (treated as fixed in this frame).

    Mercury accelerations are obtained from sixth-order spline differentiation of the Horizons
    velocity timeseries (see :func:`utils.numerical_methods.get_movements`).

    :param merc_csv: str, path to the Mercury Horizons CSV produced by data_init.py.
        Must have columns [JDTDB, X, Y, Z, VX, VY, VZ] in km / km/s / Julian days.
    :param out_file: str, output .npy path; defaults to ``config.GRAPH_FILE``.
    :param extra_bodies: list of dict, optional additional bodies, each with keys
        ``mass`` (kg) and ``csv`` (Horizons CSV in the same column convention as Mercury).
        The N-body graph becomes 2 + len(extra_bodies) nodes wide.
    :param merc_mass: float, Mercury mass (kg).
    :param sun_mass: float, Sun mass (kg).
    :param chunk_size: int, chunk size used for the JD -> decimal-year conversion.
    :param nrows: int, optional row cap applied after loading.
    :param step: int, optional row stride applied after loading.
    :return: np.ndarray of shape (T, B, 10), the saved graph snapshot tensor.
    """
    out_file = out_file if out_file is not None else config.GRAPH_FILE
    extra_bodies = extra_bodies or []

    merc = load_np(out_file.replace('.npy', '_merc_raw.npy'),
                   file_path=merc_csv,
                   nrows=nrows,
                   step=step,
                   chunk_size=chunk_size)  # [T, 7] in seconds (decimal years) and m
    merc_mov = get_movements(merc)  # [T, 10] -> [t, x, y, z, vx, vy, vz, ax, ay, az]
    T = merc_mov.shape[0]

    bodies = [merc_mov]
    masses = [merc_mass]
    for b in extra_bodies:
        b_mov = get_movements(load_np(out_file.replace('.npy', f"_{b.get('name', 'body')}_raw.npy"),
                                      file_path=b['csv'],
                                      nrows=nrows,
                                      step=step,
                                      chunk_size=chunk_size))
        if b_mov.shape[0] != T:
            raise ValueError(f"Body {b.get('name')} has {b_mov.shape[0]} timestamps, expected {T}.")
        bodies.append(b_mov)
        masses.append(b['mass'])

    B = 1 + len(bodies)  # +1 for Sun
    snapshots = np.zeros((T, B, 10), dtype=np.float64)
    snapshots[:, 0, 0] = sun_mass  # Sun mass column
    # Sun stays at origin with zero velocity / acceleration in this frame.

    for i, (mov, m) in enumerate(zip(bodies, masses), start=1):
        snapshots[:, i, 0] = m              # mass
        snapshots[:, i, 1:4] = mov[:, 1:4]  # x, y, z
        snapshots[:, i, 4:7] = mov[:, 4:7]  # vx, vy, vz
        snapshots[:, i, 7:10] = mov[:, 7:10]  # ax, ay, az

    np.save(out_file, snapshots)
    return snapshots


def load_np(data_name, file_path=None, nrows=None, step=1, reload=True, chunk_size=400000):
    """
    Loads Horizons data np file and turns into meters and decimal years
    :param data_name: str, Name of the file to save or load.
    :param file_path: str, Path to the CSV file containing the data.
    :param nrows: int, Number of rows to load.
    :param step: int, Number of steps to slice.
    :param reload: bool, Whether to reload the data. If no, save data to data_name.npy
    :param chunk_size: int, Size of each chunk for processing Julian dates.
    :return: an array containing the position, velocity, and time, vectors.
    """
    if os.path.exists(data_name) and reload:
        data = np.load(data_name)
        return data[:nrows:step]
    elif file_path:
        data_dd = dd.read_csv(file_path, dtype=np.float64).compute()

        data = data_dd.to_numpy(dtype=np.float64)

        data[:, 0] = convert_to_decimal_years(data[:, 0], chunk_size=chunk_size)

        data[:, 1:] = data[:, 1:] * 1000  # km to m (everything but time axis)

        np.save(data_name, data)
        return data[:nrows:step]
    else:
        raise Warning(f"Did/Could not reload data from {data_name} and file_path was not provided")
