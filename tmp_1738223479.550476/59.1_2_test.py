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
    # Calculate half of the angle
    half_angle = theta / 2
    
    # Compute trigonometric functions
    cos_half_angle = math.cos(half_angle)
    sin_half_angle = math.sin(half_angle)
    
    # Define rotation matrices based on the axis
    if axis == 1:  # R_x
        R = [[cos_half_angle, complex(0, -sin_half_angle)], 
             [complex(0, -sin_half_angle), cos_half_angle]]
    elif axis == 2:  # R_y
        R = [[cos_half_angle, -sin_half_angle], 
             [sin_half_angle, cos_half_angle]]
    elif axis == 3:  # R_z
        exp_neg_i_half_angle = complex(math.cos(-half_angle), math.sin(-half_angle))
        exp_i_half_angle = complex(math.cos(half_angle), math.sin(half_angle))
        R = [[exp_neg_i_half_angle, 0], 
             [0, exp_i_half_angle]]
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