from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm



# Background: In quantum mechanics and quantum computing, rotation operators are used to represent rotations 
# around specific axes in a quantum state space, typically a two-dimensional Hilbert space (qubit). 
# These operators are unitary, meaning they preserve the norm of the state vector. 
# The rotation operator around the x-axis, R_x(theta), is given by the matrix:
# R_x(theta) = [[cos(theta/2), -i*sin(theta/2)], [-i*sin(theta/2), cos(theta/2)]]
# The rotation operator around the y-axis, R_y(theta), is given by the matrix:
# R_y(theta) = [[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]]
# The rotation operator around the z-axis, R_z(theta), is given by the matrix:
# R_z(theta) = [[exp(-i*theta/2), 0], [0, exp(i*theta/2)]]
# These matrices are derived from the exponential of Pauli matrices scaled by the angle theta, 
# where the Pauli matrices are the generators of the SU(2) group.


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
    if axis == 1:  # R_x
        R = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], 
                      [-1j*np.sin(theta/2), np.cos(theta/2)]])
    elif axis == 2:  # R_y
        R = np.array([[np.cos(theta/2), -np.sin(theta/2)], 
                      [np.sin(theta/2), np.cos(theta/2)]])
    elif axis == 3:  # R_z
        R = np.array([[np.exp(-1j*theta/2), 0], 
                      [0, np.exp(1j*theta/2)]])
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