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
    # Check if the distance is beyond the cutoff
    if r >= rc:
        return 0.0
    
    # Use an array to store intermediate values
    factors = [sigma / r, sigma / rc]
    
    # Calculate sixth powers using a loop
    inv_powers = [factor ** 6 for factor in factors]
    
    # Calculate the Lennard-Jones potential using a different mathematical approach
    potential_r = epsilon * ((inv_powers[0] ** 2) - 3 * (inv_powers[0] ** 0.5))
    potential_rc = epsilon * ((inv_powers[1] ** 2) - 3 * (inv_powers[1] ** 0.5))
    
    # Return the truncated and shifted potential
    return 2 * (potential_r - potential_rc)



def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using a distinct nested loop approach with manual distance calculations.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Length of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The total potential energy of the system (in zeptojoules).
    '''
    
    def minimum_image_distance(r1, r2, L):
        # Calculate distance vector with manual wrap for minimum image
        d = [0.0] * 3
        for k in range(3):
            d[k] = r2[k] - r1[k]
            if d[k] > L / 2:
                d[k] -= L
            elif d[k] < -L / 2:
                d[k] += L
        return np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)

    def E_ij(r, sigma, epsilon, rc):
        if r >= rc:
            return 0.0
        inv_r6 = (sigma / r) ** 6
        inv_rc6 = (sigma / rc) ** 6
        potential_r = 4 * epsilon * (inv_r6 ** 2 - inv_r6)
        potential_rc = 4 * epsilon * (inv_rc6 ** 2 - inv_rc6)
        return potential_r - potential_rc

    E = 0.0
    N = len(xyz)

    # Calculate energy using nested loops with direct distance calculation
    for i in range(N - 1):
        for j in range(i + 1, N):
            r = minimum_image_distance(xyz[i], xyz[j], L)
            E += E_ij(r, sigma, epsilon, rc)

    return E




def f_ij(r_vector, sigma, epsilon, rc):
    '''Compute the Lennard-Jones force vector between two particles using a novel approach.

    Parameters:
    r_vector (array_like): The displacement vector between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.

    Returns:
    array_like: The force vector experienced by particle i due to particle j.
    '''
    # Calculate the magnitude and squared magnitude of the displacement vector
    r_mag = np.linalg.norm(r_vector)
    r_squared = r_mag ** 2
    
    # Return zero force vector if beyond cutoff distance
    if r_mag >= rc:
        return np.zeros_like(r_vector)
    
    # Precompute sigma powers to avoid redundancy
    sigma_over_r = sigma / r_mag
    sigma_over_r_2 = sigma_over_r ** 2
    sigma_over_r_6 = sigma_over_r_2 ** 3
    sigma_over_r_12 = sigma_over_r_6 ** 2
    
    # Calculate the force magnitude using a rearranged formula
    force_magnitude = 48 * epsilon * (sigma_over_r_12 - 0.5 * sigma_over_r_6) / r_squared
    
    # Use broadcasting to compute the force vector directly
    force_vector = force_magnitude * r_vector
    
    return force_vector


try:
    targets = process_hdf5_to_tuple('80.4', 3)
    target = targets[0]
    sigma = 1
    epsilon = 1
    r = np.array([-3.22883506e-03,  2.57056485e+00,  1.40822287e-04])
    rc = 1
    assert np.allclose(f_ij(r,sigma,epsilon,rc), target)

    target = targets[1]
    sigma = 2
    epsilon = 3
    r = np.array([3,  -4,  5])
    rc = 10
    assert np.allclose(f_ij(r,sigma,epsilon,rc), target)

    target = targets[2]
    sigma = 3
    epsilon = 7
    r = np.array([5,  9,  7])
    rc = 20
    assert np.allclose(f_ij(r,sigma,epsilon,rc), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e