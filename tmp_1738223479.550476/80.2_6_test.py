from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the differences in coordinates
    dx, dy, dz = r2[0] - r1[0], r2[1] - r1[1], r2[2] - r1[2]

    # Use numpy-like approach without actual numpy for minimum image convention
    def wrap(d, L):
        return (d + L / 2) % L - L / 2

    # Adjust differences
    dx, dy, dz = wrap(dx, L), wrap(dy, L), wrap(dz, L)

    # Calculate the distance using the adjusted coordinates
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return distance



def E_ij(r, sigma, epsilon, rc):
    '''Calculate the truncated and shifted Lennard-Jones potential energy between two particles.

    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.

    Returns:
    float: The potential energy between the two particles, considering the specified potentials.
    '''
    if r >= rc:
        return 0.0

    # Calculate (sigma/r)^6 using a logarithm approach to avoid direct division
    log_sigma_r = math.log(sigma) - math.log(r)
    log_sigma_rc = math.log(sigma) - math.log(rc)

    # Compute the sixth powers using exponentiation from the logarithms
    inv_r6 = math.exp(6 * log_sigma_r)
    inv_rc6 = math.exp(6 * log_sigma_rc)

    # Calculate the potential using a slightly different formula
    potential_r = 4 * epsilon * (inv_r6**2 - 2 * inv_r6)
    potential_rc = 4 * epsilon * (inv_rc6**2 - 2 * inv_rc6)

    # Return the potential energy after truncating and shifting
    return potential_r - potential_rc


try:
    targets = process_hdf5_to_tuple('80.2', 3)
    target = targets[0]
    r1 = 1.0  # Close to the sigma value
    sigma1 = 1.0
    epsilon1 = 1.0
    rc = 1
    assert np.allclose(E_ij(r1, sigma1, epsilon1, rc), target)

    target = targets[1]
    r2 = 0.5  # Significantly closer than the effective diameter
    sigma2 = 1.0
    epsilon2 = 1.0
    rc = 2
    assert np.allclose(E_ij(r2, sigma2, epsilon2, rc), target)

    target = targets[2]
    r3 = 2.0  # Larger than sigma
    sigma3 = 1.0
    epsilon3 = 1.0
    rc = 5
    assert np.allclose(E_ij(r3, sigma3, epsilon3, rc), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e