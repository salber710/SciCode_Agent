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
    
    if axis == 1:
        # Rotation around x-axis
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                      [-1j*np.sin(theta/2), np.cos(theta/2)]])
    elif axis == 2:
        # Rotation around y-axis
        R = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                      [np.sin(theta/2), np.cos(theta/2)]])
    elif axis == 3:
        # Rotation around z-axis
        R = np.array([[exp(-1j*theta/2), 0],
                      [0, exp(1j*theta/2)]])
    else:
        raise ValueError("Invalid axis value. Use 1 for x, 2 for y, or 3 for z.")

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