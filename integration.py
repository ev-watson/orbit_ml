import torch
import glob
import numpy as np
from scipy.integrate import solve_ivp
from models import InterpMLP
import config


def calculate_force(r):
    """
    Newton force
    :param r: array-like, radius in m
    :return: Radial force component
    """
    G = 6.6743e-11  # Gravitational constant in m^3 kg^-1 s^-2
    M_s = 1.988409870698051e30  # Mass of sun in kg
    m = 0.3301e24  # Mass of mercury in kg
    return -G * M_s * m / (r ** 2)


def get_L(x):
    """
    Gets theta component of angular momentum
    :param x: list of [r, phi] in that order
    :return: magnitude of the theta component of angular momentum
    """
    model = InterpMLP.load_from_checkpoint(glob.glob('*.ckpt')[0])
    v_phi_prediction = model.interp(x)
    return np.abs(x[0] * v_phi_prediction)


def ode_sys(phi, y):
    m = 0.3301e24
    r = 1 / y[0]
    v_r = y[1]
    L_theta = get_L([r, phi])
    F_r = calculate_force(r)

    dr_dphi = v_r
    dv_r_dphi = -(m * F_r) / ((L_theta * y[0]) ** 2) - y[0] / L_theta + 2 * v_r ** 2 / y[0]

    return [dr_dphi, dv_r_dphi]


def y_limit_event(phi, y):
    """
    Event function to halt the integration if y exceeds the limits
    """
    if y[0] < 1.4e-11 or y[0] > 2.25e-11:
        print(f"y_limit_event triggered at phi={phi}, y={y[0]}")
        return 0
    return 1


def dy_limit_event(phi, y):
    """
    Event function to halt the integration if dy exceeds the limits
    """
    if y[1] < -1e-14 or y[1] > 1e-14:
        print(f"dy_limit_event triggered at phi={phi}, dy={y[1]}")
        return 0
    return 1


y_limit_event.terminal = True
dy_limit_event.terminal = True

if __name__ == "__main__":
    y_0 = [1.667e-11, 0.5e-14]
    phi_range = [-np.pi, np.pi]

    solution = solve_ivp(
        ode_sys,
        phi_range,
        y_0,
        method='Radau',
        # rtol=1e-3,
        # atol=np.array([1e-15, 1e-18]),
        #first_step=0.001,
        #events=[y_limit_event, dy_limit_event],
        dense_output=True,
        t_eval=np.linspace(-np.pi+0.00001, np.pi-0.00001, 2400000),
    )

    phi_values = solution.t
    r_values = 1 / solution.y[0]
    v_r_values = solution.y[1]

    print("Phi values:", phi_values)
    print("Radial distances (r):", r_values)
    print("dy/dphi:", v_r_values)
    print(f"Solver status: {solution.status}")
    print(f"Solver message: {solution.message}")