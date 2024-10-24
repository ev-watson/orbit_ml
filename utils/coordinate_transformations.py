import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sph_to_cart_windowed(sph_coords):
    """
    Spherical to cartesian conversion for windowed data
    :param sph_coords: array-like of shape (N, 3, S), spherical position
    :return: array-like of shape (N, 3, S), cartesian position
    """
    cart = np.empty_like(sph_coords)
    for i in prange(sph_coords.shape[0]):
        r = sph_coords[i, :, 0]
        theta = sph_coords[i, :, 1]
        phi = sph_coords[i, :, 2]

        sin_theta = np.sin(theta)
        cart[i, :, 0] = r * sin_theta * np.cos(phi)
        cart[i, :, 1] = r * sin_theta * np.sin(phi)
        cart[i, :, 2] = r * np.cos(theta)
    return cart


@njit(parallel=True, fastmath=True)
def cart_to_sph_windowed(cart_coords):
    """
    Cartesian to spherical conversion for windowed data
    :param cart_coords: array-like of shape (N, S, 3), cartesian position
    :return: array-like of shape (N, S, 3), spherical position
    """
    sph = np.empty_like(cart_coords)
    for i in prange(cart_coords.shape[0]):
        x = cart_coords[i, :, 0]
        y = cart_coords[i, :, 1]
        z = cart_coords[i, :, 2]

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        sph[i, :, 0] = r
        sph[i, :, 1] = np.arccos(z / r)
        sph[i, :, 2] = np.arctan2(y, x)
    return sph


@njit
def cart_to_sph(pos):
    """
    Converts Cartesian position coordinates to Spherical position coordinates
    :param pos: array of shape (~, 3): The Cartesian position coordinates
    :return: array of shape (~, 3): Converted spherical coordinate array
    """
    N = pos.shape[0]

    sph = np.empty((N, 3), dtype=np.float64)

    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    sph[:, 0] = r
    sph[:, 1] = theta
    sph[:, 2] = phi

    return sph


@njit
def sph_to_cart(sph):
    """
    Converts Spherical position coordinates to Cartesian position coordinates.
    :param sph: array of shape (~, 3): The Spherical position coordinates
    :return: array of shape (~, 3): Converted cartesian coordinate array
    """
    N = sph.shape[0]

    cartesian = np.empty((N, 3), dtype=np.float64)

    r = sph[:, 0]
    theta = sph[:, 1]
    phi = sph[:, 2]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    cartesian[:, 0] = r * sin_theta * cos_phi
    cartesian[:, 1] = r * sin_theta * sin_phi
    cartesian[:, 2] = r * cos_theta

    return cartesian


def mean_L_vector(x=None, v=None, spherical=None, return_L=None, return_list=None):
    """
    Find mean angular momentum unit vector, if spherical coords, it returns L_theta only
    :param x: array of shape (N, 3): position vectors
    :param v: array of shape (N, 3): velocity vectors
    :param spherical: array of shape (N, 6), spherical position and velocity components (r, t, p, vr, vt, vp)
    :param return_L: anything, if not None, return average L vector unnormalized
    :param return_list: anything, if not None, return a list of L vectors
    :return: array of shape (3,) or (1,) if spherical, mean normal unit vector, if spherical, L_theta only
    """
    if spherical is not None:
        L_list = -1 * spherical[:, 0] * spherical[:, -1]
    else:
        L_list = np.cross(x, v)

    if return_list is not None:
        return L_list

    mean_L = np.mean(L_list, axis=0)

    if (return_L is not None) or (spherical is not None):
        return mean_L

    return mean_L / np.linalg.norm(mean_L)


def alignment_matrix(vec1, vec2):
    """
    For 3D vectors, returns rotation matrix that aligns vec1 with vec2
    :param vec1: array
    :param vec2: array
    :return: array, 3x3 rotation matrix that aligns vec1 with vec2
    """
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix


@njit
def apply_rotation(v, R):
    """
    Applies rotation matrix R to vector v
    :param v: array of shape (~, 3)
    :param R: array of shape (3, 3)
    :return: array of shape (~, 3)
    """
    vc = np.ascontiguousarray(v)
    Rc = np.ascontiguousarray(R)
    return np.dot(vc, Rc.T)


@njit
def reflect_across_z(v):
    """
    Reflects a (cartesian) vector, v, across the z axis, so first the xz-plane and then the yz-plane
    :param v: array-like of shape (3,), vector to be reflected
    :return: array of shape (3,), reflected vector
    """
    return np.array([-v[0], -v[1], v[2]])


def flat_plane(x, v):
    """
    Minimizes vertical variance to be in a flat plane using reflected angular momentum vector
    :param x: array of shape (~, 3), cartesian positions
    :param v: array of shape (~, 3), cartesian velocities
    :return: tuple of 2 arrays of shape (~, 3), rotated positions and velocities ((r, t, p), (vr, vt, vp))
    """
    anti_L = reflect_across_z(mean_L_vector(x, v))
    R = alignment_matrix(np.array([0, 0, 1]), anti_L)
    rotated_positions = apply_rotation(x, R)
    rotated_velocities = apply_rotation(v, R)
    return rotated_positions, rotated_velocities
