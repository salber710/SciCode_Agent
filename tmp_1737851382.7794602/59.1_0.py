import numpy as np
from cmath import exp
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.linalg import expm



# Background: In quantum mechanics, rotation operators are used to describe rotations in the state space of a quantum system. 
# The rotation operators around the x, y, and z axes are denoted as Rx, Ry, and Rz, respectively. These operators are 
# represented by 2x2 unitary matrices. The general form of these matrices for a rotation by an angle θ are:
# 
# Rx(θ) = [[cos(θ/2), -i*sin(θ/2)],
#          [-i*sin(θ/2), cos(θ/2)]]
#
# Ry(θ) = [[cos(θ/2), -sin(θ/2)],
#          [sin(θ/2), cos(θ/2)]]
#
# Rz(θ) = [[exp(-i*θ/2), 0],
#          [0, exp(i*θ/2)]]
#
# These matrices are derived from the exponential of the Pauli matrices, which are the generators of rotations in quantum mechanics.


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

from scicode.parse.parse import process_hdf5_to_tuple
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
