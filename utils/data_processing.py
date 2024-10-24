import os

import dask.dataframe as dd
import numpy as np
from astropy.time import Time
from tqdm import tqdm


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


def load_np(data_name, file_path=None, nrows=None, step=1, reload=True, chunk_size=400000):
    """
    Loads Horizons data np file and turns into meters and decimal years
    :param data_name: str, Name of the file to save or load.
    :param file_path: str, Path to the CSV file containing the data.
    :param nrows: int, Number of rows to load.
    :param step: int, Number of steps to slice.
    :param reload: bool, Whether to reload the data.
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
