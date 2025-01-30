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
    # Prepare constants using half the angle
    half_theta = theta / 2
    c = cmath.cos(half_theta)
    s = cmath.sin(half_theta)
    
    # Define a helper to create complex components
    def complex_component(real, imag):
        return real + 1j * imag

    # Use separate calculations for each axis to ensure distinct implementation
    if axis == 1:  # R_x
        R = [
            [c, complex_component(0, -s)],
            [complex_component(0, -s), c]
        ]
    elif axis == 2:  # R_y
        R = [
            [c, complex_component(-s, 0)],
            [complex_component(s, 0), c]
        ]
    elif axis == 3:  # R_z
        exp_neg = complex_component(cmath.cos(-half_theta), cmath.sin(-half_theta))
        exp_pos = complex_component(cmath.cos(half_theta), cmath.sin(half_theta))
        R = [
            [exp_neg, 0],
            [0, exp_pos]
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