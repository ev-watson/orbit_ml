import argparse
import os
import warnings

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Data from APL requests")
parser.add_argument("--object", "-o", type=str, default="merc")
parser.add_argument("--center-yr", '-cy', type=int, default=1870)
parser.add_argument("--length", '-l', type=int, default=40)
parser.add_argument("--name", '-n', type=str, default="horizons")
args = parser.parse_args()


def fetch_data(year, jump, orbital_elements, earth):
    if orbital_elements:
        url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='199'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='ELEMENTS'&CENTER='500@10'&START_TIME='{year:0>4}-Apr-01 00:00 TDB'&STOP_TIME='{year + jump:0>4}-Apr-01 00:00'&STEP_SIZE='22505'&REF_SYSTEM='ICRF'&REF_PLANE='FRAME'&CSV_FORMAT='YES'"
        usecols = [0, 2, 3, 4, 5, 6, 7, 11, 13]
        colnames = ['JDTDB', 'EC', 'QR', 'IN', 'OM', 'W', 'Tp', 'A', 'PR']
        skiprows = 51
        skipfooter = 72
        name = 'ooe'
    elif earth:
        url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='399'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&CENTER='500@10'&START_TIME='{year:0>4}-Apr-01 00:00 TDB'&STOP_TIME='{year + jump:0>4}-Apr-01 00:00'&STEP_SIZE='12m'&REF_SYSTEM='ICRF'&REF_PLANE='FRAME'&VEC_TABLE='1'&CSV_FORMAT='YES'"
        usecols = [0, 2, 3, 4]
        colnames = ['JDTDB', 'X', 'Y', 'Z']
        skiprows = 57
        skipfooter = 63
        name = 'earth'
    else:
        url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='199'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&CENTER='500@10'&START_TIME='{year:0>4}-Apr-01 00:00 TDB'&STOP_TIME='{year + jump:0>4}-Apr-01 00:00'&STEP_SIZE='12m'&REF_SYSTEM='ICRF'&REF_PLANE='FRAME'&VEC_TABLE='2'&CSV_FORMAT='YES'"
        usecols = [0, 2, 3, 4, 5, 6, 7]
        colnames = ['JDTDB', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
        skiprows = 50
        skipfooter = 66
        name = 'merc'

    request = requests.get(url)
    results_name = f"results_{name}.txt"

    if request.status_code == 200:
        with open(results_name, 'w') as results:
            results.write(request.text)

        good = False
        count = 0
        while not good and count < 2:
            try:
                df = pd.read_csv(results_name,
                                 usecols=usecols,
                                 names=colnames,
                                 skiprows=skiprows,
                                 skipfooter=skipfooter,
                                 engine='python', dtype=np.float64)
                good = True
            except Exception as e:
                error_name = f"results_{name}_error.txt"
                warnings.warn(f"Bad text at year {year}: {str(e)}, copying bad text to {error_name}")
                os.system(f"cp {results_name} {error_name}")
                warnings.warn(f"Saved file to {error_name}, cleaning text of Null Bytes...")
                clean_text = request.text.replace("\0", "")
                with open(results_name, 'w') as results:
                    results.write(clean_text)
                count += 1

        if count == 2:
            raise SyntaxError("2 attempts made at cleaning text, quitting...")

        # noinspection PyUnboundLocalVariable
        return df

    else:
        warning_str = f"Request failed with status code {request.status_code}"
        if request.status_code == 400:
            raise Exception(f"{warning_str}: Bad request")
        elif request.status_code == 405:
            raise Exception(f"{warning_str}: Not allowed")
        elif request.status_code == 500:
            raise Exception(f"{warning_str}: Server error")
        elif request.status_code == 503:
            raise Exception(f"{warning_str}: Service unavailable")
        else:
            raise Exception(warning_str)


def get_horizons(length, center_yr=2012, time_step='lowest', orbital_elements=False, earth=False):
    if orbital_elements:
        if time_step == 'lowest':
            jump = 1
        elif isinstance(time_step, int):
            jump = time_step
        else:
            raise ValueError('Invalid time step')
    else:
        if time_step == 'lowest':
            jump = 2
        elif time_step == 'half hour':
            jump = 5
        elif time_step == 'hour':
            jump = 10
        elif isinstance(time_step, int):
            jump = time_step
        else:
            raise ValueError('Invalid time step')

    start = center_yr
    years = [i + start for i in range(-length // 2, length // 2, jump)]

    dfs = []
    for year in tqdm(years, desc="Fetching data"):
        df = fetch_data(year, jump, orbital_elements, earth)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df


if __name__ == '__main__':
    # Fetch orbital elements data
    # OOE = get_horizons(1000, time_step='lowest', orbital_elements=True)
    # OOE.to_csv('OOE.csv', index=False)

    # Fetch Earth data
    if args.object == 'earth':
        horizons = get_horizons(args.length, center_yr=args.center_yr, time_step='lowest', earth=True)
        horizons.to_csv('earth_pos.csv', index=False)

    # Fetch Mercury data
    if args.object == 'merc':
        horizons = get_horizons(args.length, center_yr=args.center_yr, time_step='lowest')
        horizons.to_csv(f"{args.name}.csv", index=False)
