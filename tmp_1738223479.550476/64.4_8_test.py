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
    # Calculate the ratio of r to sigma
    r_over_sigma = r / sigma
    
    # Use reciprocal powers to compute the sixth and twelfth terms
    r_over_sigma_6_inv = (sigma / r)**6
    r_over_sigma_12_inv = r_over_sigma_6_inv * r_over_sigma_6_inv
    
    # Calculate the Lennard-Jones potential using the reciprocal terms
    E_lj = 4 * epsilon * (r_over_sigma_12_inv - r_over_sigma_6_inv)
    
    return E_lj




def E_i(r, positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array_like
        The (x, y, z) coordinates of the target particle.
    positions : array_like
        An array of (x, y, z) coordinates for each of the other particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''

    def distance_periodic(r1, r2, L):
        '''Calculate distance considering periodic boundary conditions using a vectorized approach.'''
        delta = (r1 - r2) - L * np.round((r1 - r2) / L)
        return np.linalg.norm(delta)

    def lj_energy(r, sigma, epsilon):
        '''Calculate Lennard-Jones potential energy using a compact formulation.'''
        sr_ratio = sigma / r
        sr_ratio_6 = sr_ratio ** 6
        sr_ratio_12 = sr_ratio_6 * sr_ratio_6
        return 4 * epsilon * (sr_ratio_12 - sr_ratio_6)

    energies = [
        lj_energy(distance_periodic(r, pos, L), sigma, epsilon)
        for pos in positions if not (np.isclose(r, pos).all())
    ]
    
    return np.sum(energies)


try:
    targets = process_hdf5_to_tuple('64.4', 4)
    target = targets[0]
    r1 = np.array([1, 1, 1])
    positions1 = np.array([[9, 9, 9], [5, 5, 5]])
    L1 = 10.0
    sigma1 = 1.0
    epsilon1 = 1.0
    assert np.allclose(E_i(r1, positions1, L1, sigma1, epsilon1), target)

    target = targets[1]
    r2 = np.array([5, 5, 5])
    positions2 = np.array([[5.1, 5.1, 5.1], [4.9, 4.9, 4.9], [5, 5, 6]])
    L2 = 10.0
    sigma2 = 1.0
    epsilon2 = 1.0
    assert np.allclose(E_i(r2, positions2, L2, sigma2, epsilon2), target)

    target = targets[2]
    r3 = np.array([0.1, 0.1, 0.1])
    positions3 = np.array([[9.9, 9.9, 9.9], [0.2, 0.2, 0.2]])
    L3 = 10.0
    sigma3 = 1.0
    epsilon3 = 1.0
    assert np.allclose(E_i(r3, positions3, L3, sigma3, epsilon3), target)

    target = targets[3]
    r3 = np.array([1e-8, 1e-8, 1e-8])
    positions3 = np.array([[1e-8, 1e-8, 1e-8], [1e-8, 1e-8, 1e-8]])
    L3 = 10.0
    sigma3 = 1.0
    epsilon3 = 1.0
    assert np.allclose(E_i(r3, positions3, L3, sigma3, epsilon3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e