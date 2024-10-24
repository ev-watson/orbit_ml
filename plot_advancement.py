import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from astropy.time import Time
import vg
from tqdm import tqdm
from utils import mean_L_vector
from dask.diagnostics import ProgressBar


def quadplusperiodic(t, A, w, Q):
    """
    Fit function for Mercury's orbit adopted from R. S. Park et al., Astronomical Journal volume 153, paper 121 (7 pages), 2017.

    :param t: time
    :param A: constant offset
    :param w: perihelion advancement rate
    :param Q: perihelion advancement acceleration
    :return: advancement
    """
    # Define mean motion of all relevant (periodic frequencies that have larger than half arcsec effect) planets
    nM = 2 * np.pi / 87.969257
    nV = 2 * np.pi / 224.701
    nE = 2 * np.pi / 365.25
    nJ = 2 * np.pi / 4332.59
    nS = 2 * np.pi / 10755.70
    frequencies = [2 * nV, nV, nM - 2 * nV, 2 * nM - 3 * nV, nM - 3 * nV, 2 * nM - 4 * nV, 2 * nM - 5 * nV, nM - 2 * nE,
                   nM - 4 * nE, 3 * nJ, 2 * nJ, nJ, nM - 2 * nJ, 2 * nS]
    # Sine coefficients
    S = [-0.44, 0.19, -3.67, -0.55, 1.93, 0.25, 3.37, 0.46, -0.21, 0.42, 0.39, -0.55, 0.16, 0.5]

    # Cosine coefficients
    C = [-0.24, -0.70, 2.55, -0.38, 1.55, -0.78, 0.05, -0.61, -0.54, -0.63, -7.23, 1.44, 0.82, -0.71]

    total = 0
    for frequency, Si, Ci in zip(frequencies, S, C):
        total += Si * np.sin(frequency * t) + Ci * np.cos(frequency * t)

    return A + w * t + Q * t ** 2 + total


def get_peri_indices(ds):
    """
    Grabs periapsis vector indices from a list of distances from the center coord in an elliptical orbit.

    :param ds: array-like of distances from center coord.
    :return: list of indices of peri-vectors
    """
    peri_indices = []
    ds = ds.compute()
    for i in tqdm(range(1, len(ds) - 1), desc="Finding Periapsis Indices"):
        if ds.iloc[i - 1] > ds.iloc[i] < ds.iloc[i + 1]:
            peri_indices.append(i)
    return peri_indices


def plot_advancement(data, perihelion_indices=None, plot=True):
    """
    Plots advancement over time from ephemeral data.

    :param plot: bool, whether or not to plot
    :param data: pd.DataFrame of Mercury's ephemeral data
    :param perihelion_indices: list of indices of perihelion vectors
    :return: Plot of perihelion longitudinal advancement over time and x and y list
    """
    position_column_names = ['X', 'Y', 'Z']

    if perihelion_indices is None:
        peri_indices = get_peri_indices(data['RG'])
    else:
        peri_indices = perihelion_indices

    n = len(data)
    date_time = Time(data['JDTDB'].compute(), format='jd', scale='tdb')
    date_time.format = 'decimalyear'
    dates = dd.from_pandas(pd.Series([t.decimalyear for t in date_time], dtype=np.float64), npartitions=n)
    dates = dates.compute()
    ref_year = dates.iloc[len(dates) // 2 - 1]  # make reference date the middle
    adv_list = []
    adv_times = []

    reference_peri = np.array(data.compute().iloc[peri_indices[len(peri_indices) // 2]][position_column_names])

    mean_orbit_pole_vector = mean_L_vector(data.iloc[n // 2 - 440:n // 2 + 440])

    for p in tqdm(peri_indices, desc="Calculating Advancements"):
        peri = np.array(data.compute().iloc[p][position_column_names])

        advancement = vg.signed_angle(reference_peri, peri, look=mean_orbit_pole_vector) * 3600

        adv_list.append(advancement)
        adv_times.append(dates.iloc[p] - dates.iloc[n // 2])  # in years

    if plot:
        fig, ax = plt.subplots()
        ax.set_title("Precession of Mercury's orbit over time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mercury Perihelion advancement (arcsec)")

        ax.set_xticks([adv_times[0], 0, adv_times[-1]], labels=[f'{ref_year + adv_times[0] * 100:.0f}', f'{ref_year}',
                                                                f'{ref_year + adv_times[-1] * 100:.0f}'])
        ax.plot(adv_times, adv_list)

        params, covar = curve_fit(quadplusperiodic, adv_times, adv_list)
        fig.suptitle(f"w: {params[1]:.3f} as/yr, Q: {params[2]:.4f} as/yr^2")
        plt.show()
        plt.savefig(f"plot_advancement_{ref_year:.0f}.png", dpi=300)
        plt.close()

    return np.array(adv_times), np.array(adv_list)


if __name__ == '__main__':
    with ProgressBar():
        df = dd.read_csv('horizons.csv', dtype=float, blocksize=300e6)
        x_data, y_data = plot_advancement(df)
        np.save('x_data.npy', x_data)
        np.save('y_data.npy', y_data)
        parameters, covariance = curve_fit(quadplusperiodic, x_data, y_data)
        print(f'w: {parameters[1]} as/yr, Q: {parameters[2]} as/yr^2')
