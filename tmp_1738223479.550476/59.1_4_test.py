from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm




def rotation_matrices(axis, theta):
    '''Create rotation matrices Rx, Ry, and Rz with the given angle theta.
    Inputs:
    axis : int
        The rotation axis. 1 = x, 2 = y, 3 = z.
    theta : float
        The rotation angle.
    Output:
    R : matrix of shape(2, 2)
        The rotation matrix.
    '''
    # Precompute constants for efficiency
    half_theta = theta / 2
    cos_half_theta = cmath.cos(half_theta)
    sin_half_theta = cmath.sin(half_theta)
    i_sin_half_theta = 1j * sin_half_theta
    exp_half_theta_neg = cmath.exp(-1j * half_theta)
    exp_half_theta_pos = cmath.exp(1j * half_theta)

    # Use dictionary to store the rotation matrices for each axis
    matrices = {
        1: [[cos_half_theta, -i_sin_half_theta],
            [-i_sin_half_theta, cos_half_theta]],
        2: [[cos_half_theta, -sin_half_theta],
            [sin_half_theta, cos_half_theta]],
        3: [[exp_half_theta_neg, 0],
            [0, exp_half_theta_pos]]
    }

    # Validate axis and return the corresponding rotation matrix
    if axis not in matrices:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    
    return matrices[axis]


try:
    targets = process_hdf5_to_tuple('59.1', 3)
    target = targets[0]
    axis = 1
    theta = np.pi
    assert np.allclose(1j * rotation_matrices(axis, theta), target)

    target = targets[1]
    axis = 2
    theta = np.pi
    assert np.allclose(1j * rotation_matrices(axis, theta), target)

    target = targets[2]
    axis = 3
    theta = np.pi
    assert np.allclose(1j * rotation_matrices(axis, theta), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e