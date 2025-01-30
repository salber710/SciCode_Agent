from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Convert r to a numpy array if it's not already
    r = np.array(r, dtype=float)
    
    # Use numpy's clip function to wrap coordinates
    # This method first shifts the coordinates to positive by adding L to negative values,
    # and then uses np.clip to ensure they stay within the bounds [0, L)
    shifted_r = np.where(r < 0, r + L, r)
    coord = (shifted_r - np.floor(shifted_r / L) * L)
    
    # Clip to ensure the coordinates are in the [0, L) interval
    coord = np.clip(coord, 0, L - np.finfo(float).eps)
    
    return coord


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    
    # Using a generator expression with map to calculate the minimum image distances
    diff = map(lambda x, y: x - y, r1, r2)
    
    def min_image(d, L):
        half_L = L / 2
        if d > half_L:
            return d - L
        elif d < -half_L:
            return d + L
        return d

    # Calculate the adjusted differences using the minimum image convention
    adjusted_diff = map(lambda d: min_image(d, L), diff)

    # Calculate and return the Euclidean distance
    distance = sum(d**2 for d in adjusted_diff) ** 0.5
    return distance



def E_ij(r, sigma, epsilon):
    '''Calculate the Lennard-Jones potential energy between two particles.
    Parameters:
    r : float
        The distance between the two particles.
    sigma : float
        The distance at which the potential minimum occurs.
    epsilon : float
        The depth of the potential well.
    Returns:
    float
        The potential energy between the two particles at distance r.
    '''
    # Calculate the distance ratio (r/sigma) and its reciprocal
    distance_ratio = r / sigma
    reciprocal_ratio = sigma / r
    
    # Compute powers of the reciprocal ratio
    r6 = reciprocal_ratio**6
    r12 = r6**2
    
    # Use intermediate variables to calculate the Lennard-Jones potential
    attractive_part = 4 * epsilon * r6
    repulsive_part = 4 * epsilon * r12
    
    # Calculate the Lennard-Jones potential by subtracting the attractive part from the repulsive part
    E_lj = repulsive_part - attractive_part
    
    return E_lj


try:
    targets = process_hdf5_to_tuple('64.3', 3)
    target = targets[0]
    r1 = 1.0  # Close to the sigma value
    sigma1 = 1.0
    epsilon1 = 1.0
    assert np.allclose(E_ij(r1, sigma1, epsilon1), target)  # Expected to be 0, as it's at the potential minimum

    target = targets[1]
    r2 = 0.5  # Significantly closer than the effective diameter
    sigma2 = 1.0
    epsilon2 = 1.0
    assert np.allclose(E_ij(r2, sigma2, epsilon2), target)

    target = targets[2]
    r3 = 2.0  # Larger than sigma
    sigma3 = 1.0
    epsilon3 = 1.0
    assert np.allclose(E_ij(r3, sigma3, epsilon3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e