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

    def wrapped_distance(r1, r2, box_length):
        '''Calculate wrapped distance using an alternative approach.'''
        delta = np.mod(r1 - r2 + box_length / 2, box_length) - box_length / 2
        return np.sqrt(np.sum(delta ** 2))

    def compute_lj(r, sigma, epsilon):
        '''Compute Lennard-Jones potential energy with a different method.'''
        r_inv = sigma / r
        r_inv_6 = r_inv ** 6
        return 4 * epsilon * (r_inv_6 * r_inv_6 - r_inv_6)

    energy_sum = np.sum([
        compute_lj(wrapped_distance(r, pos, L), sigma, epsilon)
        for pos in positions if not np.allclose(r, pos)
    ])

    return energy_sum




def E_system(positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a system of particles in a periodic system.
    Parameters:
    positions : array_like
        An array of (x, y, z) coordinates for each of the particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the system.
    '''
    
    def compute_minimum_image_distance(r1, r2, box_length):
        '''Calculate the minimum image distance between two points in periodic boundary conditions.'''
        delta = r1 - r2
        delta -= box_length * np.round(delta / box_length)
        return np.linalg.norm(delta)

    def lennard_jones_potential(r, sigma, epsilon):
        '''Calculate the Lennard-Jones potential.'''
        if r == 0:
            return 0  # Avoid division by zero
        r_inv = sigma / r
        r_inv_6 = r_inv ** 6
        return 4 * epsilon * (r_inv_6 ** 2 - r_inv_6)

    num_particles = len(positions)
    total_energy = 0.0
    
    # Use a nested loop to iterate over unique pairs
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            distance = compute_minimum_image_distance(positions[i], positions[j], L)
            total_energy += lennard_jones_potential(distance, sigma, epsilon)
    
    return total_energy


try:
    targets = process_hdf5_to_tuple('64.5', 3)
    target = targets[0]
    positions1 = np.array([[1, 1, 1], [1.1, 1.1, 1.1]])
    L1 = 10.0
    sigma1 = 1.0
    epsilon1 = 1.0
    assert np.allclose(E_system(positions1, L1, sigma1, epsilon1), target)

    target = targets[1]
    positions2 = np.array([[1, 1, 1], [1, 9, 1], [9, 1, 1], [9, 9, 1]])
    L2 = 10.0
    sigma2 = 1.0
    epsilon2 = 1.0
    assert np.allclose(E_system(positions2, L2, sigma2, epsilon2), target)

    target = targets[2]
    np.random.seed(0)
    positions3 = np.random.rand(10, 3) * 10  # 10 particles in a 10x10x10 box
    L3 = 10.0
    sigma3 = 1.0
    epsilon3 = 1.0
    assert np.allclose(E_system(positions3, L3, sigma3, epsilon3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e