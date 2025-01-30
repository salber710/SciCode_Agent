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
    # Define sine and cosine of half theta
    half_theta = theta / 2
    cos_half_theta = cmath.cos(half_theta)
    sin_half_theta = cmath.sin(half_theta)

    # Define imaginary unit
    I = complex(0, 1)

    # Create rotation matrices using complex exponentials directly
    if axis == 1:  # R_x
        R = [
            [cos_half_theta, -I * sin_half_theta],
            [-I * sin_half_theta, cos_half_theta]
        ]
    elif axis == 2:  # R_y
        R = [
            [cos_half_theta, -sin_half_theta],
            [sin_half_theta, cos_half_theta]
        ]
    elif axis == 3:  # R_z
        exp_neg_i_half_theta = cmath.exp(-I * half_theta)
        exp_i_half_theta = cmath.exp(I * half_theta)
        R = [
            [exp_neg_i_half_theta, 0],
            [0, exp_i_half_theta]
        ]
    else:
        raise ValueError("Axis must be 1 (x), 2 (y), or 3 (z).")
    
    return R


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