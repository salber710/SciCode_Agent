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



# Background: The Lennard-Jones potential is a mathematical model to approximate the interaction between a pair of neutral atoms or molecules. 
# The potential is defined by two parameters: sigma (σ), which is the distance at which the potential is zero, and epsilon (ε), which is the depth of the potential well.
# The potential is given by the formula:
#     V(r) = 4ε [(σ/r)^12 - (σ/r)^6]
# The potential is often truncated and shifted at a cutoff distance rc to improve computational efficiency and to avoid calculating forces at long distances where the potential is negligible.
# The truncated and shifted potential is defined such that V(rc) = 0 for r >= rc. This is done by computing:
#     V_shifted(r) = V(r) - V(rc) for r < rc
#     V_shifted(r) = 0 for r >= rc


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
    if r < rc:
        # Calculate the Lennard-Jones potential
        lj_potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        # Calculate the potential at the cutoff distance
        lj_potential_rc = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
        # Return the truncated and shifted potential
        return lj_potential - lj_potential_rc
    else:
        return 0.0


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