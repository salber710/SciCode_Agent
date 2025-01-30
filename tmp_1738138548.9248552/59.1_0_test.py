from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm



# Background: In quantum mechanics, rotation operators are used to describe rotations in the state space of a quantum system. 
# The rotation operators around the x, y, and z axes are denoted as Rx, Ry, and Rz, respectively. These operators are 
# represented by 2x2 matrices and are parameterized by an angle θ. The matrices are derived from the Pauli matrices, 
# which are fundamental in quantum mechanics. The rotation matrices are given by:
# Rx(θ) = exp(-i * θ/2 * σx) = [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
# Ry(θ) = exp(-i * θ/2 * σy) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
# Rz(θ) = exp(-i * θ/2 * σz) = [[exp(-i*θ/2), 0], [0, exp(i*θ/2)]]
# where σx, σy, and σz are the Pauli matrices:
# σx = [[0, 1], [1, 0]], σy = [[0, -i], [i, 0]], σz = [[1, 0], [0, -1]].


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
    if axis == 1:  # Rx
        R = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                      [-1j * np.sin(theta / 2), np.cos(theta / 2)]])
    elif axis == 2:  # Ry
        R = np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                      [np.sin(theta / 2), np.cos(theta / 2)]])
    elif axis == 3:  # Rz
        R = np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]])
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