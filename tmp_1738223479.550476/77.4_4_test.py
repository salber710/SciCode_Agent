from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use a while loop to iteratively adjust each coordinate to lie within the range [0, L).
    wrapped_coords = []
    for ri in r:
        while ri >= L:
            ri -= L
        while ri < 0:
            ri += L
        wrapped_coords.append(ri)
    return wrapped_coords


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    
    # Calculate the difference vector between r2 and r1
    delta = [(r2[i] - r1[i]) for i in range(3)]
    
    # Apply the minimum image convention by checking the shortest distance directly
    for i in range(3):
        if delta[i] > L / 2:
            delta[i] -= L
        elif delta[i] <= -L / 2:
            delta[i] += L
    
    # Compute and return the Euclidean distance using a generator expression
    return sum((d ** 2 for d in delta)) ** 0.5


def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    r12: tuple of floats, the minimum image vector between the two atoms.
    '''
    # Calculate the difference vector and apply minimum image convention using a mathematical function
    def min_image_component(d, L):
        # Use mathematical expression to adjust components
        return d - L * round(d / L)

    # Calculate each component using the helper function
    delta_x = min_image_component(r2[0] - r1[0], L)
    delta_y = min_image_component(r2[1] - r1[1], L)
    delta_z = min_image_component(r2[2] - r1[2], L)

    # Return the minimum image vector as a tuple
    return (delta_x, delta_y, delta_z)




def E_ij(r, sigma, epsilon, rc):
    '''Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The combined potential energy between the two particles, considering the specified potentials.
    '''
    if r >= rc:
        return 0.0
    
    # Use an alternative approach with a focus on dividing the computation into distinct steps
    def lj_energy(distance, sig, eps):
        # Calculate the reciprocal of the distance and its powers
        inv_dist = sig / distance
        inv_dist_2 = inv_dist * inv_dist
        inv_dist_4 = inv_dist_2 * inv_dist_2
        inv_dist_8 = inv_dist_4 * inv_dist_4
        inv_dist_12 = inv_dist_8 * inv_dist_4
        
        # Calculate the Lennard-Jones potential
        potential = 4 * eps * (inv_dist_12 - inv_dist_4 * inv_dist_2)
        return potential
    
    # Calculate the potential at r and rc using the new approach
    potential_r = lj_energy(r, sigma, epsilon)
    potential_rc = lj_energy(rc, sigma, epsilon)
    
    # Return the calculated truncated and shifted potential
    return potential_r - potential_rc


try:
    targets = process_hdf5_to_tuple('77.4', 3)
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
    rc = 3
    assert np.allclose(E_ij(r3, sigma3, epsilon3, rc), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e