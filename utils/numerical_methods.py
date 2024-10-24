import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def socfdw(x, dt=1.0, dtype=np.float64):
    """
    Sixth-order central difference for windowed arrays with shape [N, s, f],
    utilizing Numba for acceleration. End points are treated with special stencils
    to achieve 6th order accuracy.

    :param x: np.ndarray of shape (N, s, f), where
              N = number of samples,
              f = number of features,
              s = sequence length.
    :param dt: float, time step. Default is 1.0.
    :param dtype: np.dtype to use.
    :return: np.ndarray of shape (N, s, f), representing the numerical derivative dx/dt.
    """
    N, s, f = x.shape

    if s < 7:
        raise ValueError("Sequence length must be at least 7 for sixth-order central difference.")

    # Initialize the output array with zeros
    dxdt = np.zeros((N, s, f), dtype=dtype)

    # Central difference coefficient
    central_coeff = 1.0 / (60.0 * dt)

    # Define forward coefficients for boundary points (seq=0,1,2)
    forward_coeff_first = np.array([-147.0, 360.0, -450.0, 400.0, -225.0, 72.0, -10.0], dtype=dtype) * central_coeff
    forward_coeff_second = np.array([-10.0, -77.0, 150.0, -100.0, 50.0, -15.0, 2.0], dtype=dtype) * central_coeff
    forward_coeff_third = np.array([2.0, -24.0, -35.0, 80.0, -30.0, 8.0, -1.0], dtype=dtype) * central_coeff

    # Define backward coefficients for boundary points (seq=s-3, s-2, s-1)
    backward_coeff_third_last = np.array([1.0, -8.0, 30.0, -80.0, 35.0, 24.0, -2.0], dtype=dtype) * central_coeff
    backward_coeff_second_last = np.array([-2.0, 15.0, -50.0, 100.0, -150.0, 77.0, 10.0], dtype=dtype) * central_coeff
    backward_coeff_last = np.array([10.0, -72.0, 225.0, -400.0, 450.0, -360.0, 147.0], dtype=dtype) * central_coeff

    for n in prange(N):
        for feat in range(f):
            # Compute central differences for interior points
            dxdt[n, 3:-3, feat] = central_coeff * (
                    -x[n, :-6, feat] +
                    9.0 * x[n, 1:-5, feat] -
                    45.0 * x[n, 2:-4, feat] +
                    45.0 * x[n, 4:-2, feat] -
                    9.0 * x[n, 5:-1, feat] +
                    x[n, 6:, feat]
            )

            # Apply forward stencils for seq=0,1,2
            first_boundaries = x[n, :7, feat]
            dxdt[n, 0, feat] = (
                first_boundaries[0] * forward_coeff_first[0] +
                first_boundaries[1] * forward_coeff_first[1] +
                first_boundaries[2] * forward_coeff_first[2] +
                first_boundaries[3] * forward_coeff_first[3] +
                first_boundaries[4] * forward_coeff_first[4] +
                first_boundaries[5] * forward_coeff_first[5] +
                first_boundaries[6] * forward_coeff_first[6]
            )
            dxdt[n, 1, feat] = (
                first_boundaries[0] * forward_coeff_second[0] +
                first_boundaries[1] * forward_coeff_second[1] +
                first_boundaries[2] * forward_coeff_second[2] +
                first_boundaries[3] * forward_coeff_second[3] +
                first_boundaries[4] * forward_coeff_second[4] +
                first_boundaries[5] * forward_coeff_second[5] +
                first_boundaries[6] * forward_coeff_second[6]
            )
            dxdt[n, 2, feat] = (
                first_boundaries[0] * forward_coeff_third[0] +
                first_boundaries[1] * forward_coeff_third[1] +
                first_boundaries[2] * forward_coeff_third[2] +
                first_boundaries[3] * forward_coeff_third[3] +
                first_boundaries[4] * forward_coeff_third[4] +
                first_boundaries[5] * forward_coeff_third[5] +
                first_boundaries[6] * forward_coeff_third[6]
            )

            # Apply backward stencils for seq=s-3, s-2, s-1
            last_boundaries = x[n, -7:, feat]
            dxdt[n, -3, feat] = (
                last_boundaries[0] * backward_coeff_third_last[0] +
                last_boundaries[1] * backward_coeff_third_last[1] +
                last_boundaries[2] * backward_coeff_third_last[2] +
                last_boundaries[3] * backward_coeff_third_last[3] +
                last_boundaries[4] * backward_coeff_third_last[4] +
                last_boundaries[5] * backward_coeff_third_last[5] +
                last_boundaries[6] * backward_coeff_third_last[6]
            )
            dxdt[n, -2, feat] = (
                last_boundaries[0] * backward_coeff_second_last[0] +
                last_boundaries[1] * backward_coeff_second_last[1] +
                last_boundaries[2] * backward_coeff_second_last[2] +
                last_boundaries[3] * backward_coeff_second_last[3] +
                last_boundaries[4] * backward_coeff_second_last[4] +
                last_boundaries[5] * backward_coeff_second_last[5] +
                last_boundaries[6] * backward_coeff_second_last[6]
            )
            dxdt[n, -1, feat] = (
                last_boundaries[0] * backward_coeff_last[0] +
                last_boundaries[1] * backward_coeff_last[1] +
                last_boundaries[2] * backward_coeff_last[2] +
                last_boundaries[3] * backward_coeff_last[3] +
                last_boundaries[4] * backward_coeff_last[4] +
                last_boundaries[5] * backward_coeff_last[5] +
                last_boundaries[6] * backward_coeff_last[6]
            )

    return dxdt


@njit
def sixth_order_central_difference(x, dt):
    """
    Sixth order central difference, with end points being treated with special stencils to 6th order accuracy.
    :param x: array of shape (N, F): 1/2D array of values.
    :param dt: float, time step.
    :return: array of shape (N, F): dxdt to sixth order accuracy.
    """
    N = x.shape[0]

    if N < 7:
        raise ValueError("Input array must have at least 7 points for sixth-order central difference.")

    dxdt = np.zeros_like(x)

    # Central difference for interior points
    central_coeff = 1 / (60 * dt)
    dxdt[3:-3] = central_coeff * (-x[:-6] + 9 * x[1:-5] - 45 * x[2:-4] + 45 * x[4:-2] - 9 * x[5:-1] + x[6:])

    # Special stencils for boundary points made from wikipedia

    first_boundaries = np.ascontiguousarray(x[:7])
    last_boundaries = np.ascontiguousarray(x[-7:])
    # First point: Normal forward difference
    forward_coeff_first = central_coeff * np.array([-147, 360, -450, 400, -225, 72, -10])
    dxdt[0] = np.dot(forward_coeff_first, first_boundaries)

    # Second point: Including the previous point
    forward_coeff_second = central_coeff * np.array([-10, -77, 150, -100, 50, -15, 2])
    dxdt[1] = np.dot(forward_coeff_second, first_boundaries)

    # Third point: Including the previous two points
    forward_coeff_third = central_coeff * np.array([2, -24, -35, 80, -30, 8, -1])
    dxdt[2] = np.dot(forward_coeff_third, first_boundaries)

    # Third-to-last point: Including the next (last) two points
    backward_coeff_third_last = central_coeff * np.array([1, -8, 30, -80, 35, 24, -2])
    dxdt[-3] = np.dot(backward_coeff_third_last, last_boundaries)

    # Second-to-last point: Including the next (last) point
    backward_coeff_second_last = central_coeff * np.array([-2, 15, -50, 100, -150, 77, 10])
    dxdt[-2] = np.dot(backward_coeff_second_last, last_boundaries)

    # Last point: Normal backward difference
    backward_coeff_last = central_coeff * np.array([10, -72, 225, -400, 450, -360, 147])
    dxdt[-1] = np.dot(backward_coeff_last, last_boundaries)

    return dxdt
