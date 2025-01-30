import numpy as np
import itertools

# Background: In computational simulations, especially those involving particles in a confined space, 
# it is common to use periodic boundary conditions (PBCs) to simulate an infinite system using a finite 
# simulation box. This is done by wrapping the coordinates of particles such that when a particle moves 
# out of one side of the box, it re-enters from the opposite side. This helps in avoiding edge effects 
# and mimics a larger, continuous space. The wrapping is typically done by taking the modulus of the 
# particle's position with respect to the box size. For a cubic box of size L, the wrapped coordinate 
# can be calculated as r' = r % L, where r is the original coordinate.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    if L <= 0:
        raise ValueError("Box size L must be positive.")
    if not isinstance(r, np.ndarray):
        raise TypeError("Input r must be a numpy array.")
    if r.ndim != 1 or r.size != 3:
        raise ValueError("Input r must be a 1D array with three elements.")
    if not np.issubdtype(r.dtype, np.number):
        raise TypeError("Elements of r must be numeric.")
    # Use numpy's modulus operation to wrap the coordinates
    coord = np.mod(r, L)
    return coord


# Background: In computational simulations of particles within a periodic cubic system, the concept of 
# minimum image distance is crucial for calculating interactions between particles. The minimum image 
# distance is the shortest distance between two particles considering the periodic boundary conditions. 
# In a cubic box of size L, each particle can be thought of as having infinite periodic images in all 
# directions. The minimum image convention ensures that we consider the closest image of a particle 
# when calculating distances. This is done by adjusting the distance between two particles such that 
# it is within half the box length in each dimension. Mathematically, for each dimension, the distance 
# is adjusted using the formula: d = r2 - r1, and then d = d - L * round(d / L), where round is the 
# nearest integer function. This ensures that the distance is the shortest possible considering the 
# periodic boundaries.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    if L <= 0:
        raise ValueError("Box size L must be positive.")
    if not isinstance(r1, np.ndarray) or not isinstance(r2, np.ndarray):
        raise TypeError("Inputs r1 and r2 must be numpy arrays.")
    if r1.ndim != 1 or r1.size != 3 or r2.ndim != 1 or r2.size != 3:
        raise ValueError("Inputs r1 and r2 must be 1D arrays with three elements.")
    if not np.issubdtype(r1.dtype, np.number) or not np.issubdtype(r2.dtype, np.number):
        raise TypeError("Elements of r1 and r2 must be numeric.")
    
    # Calculate the vector difference
    delta = r2 - r1
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    # Ensure the distance is calculated correctly when close to box boundaries
    delta = np.where(np.abs(delta) > L / 2, delta - np.sign(delta) * L, delta)
    
    # Calculate the Euclidean distance
    distance = np.sqrt(np.sum(delta**2))
    
    return distance


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model the forces between particles. The potential is characterized by two parameters: 
# sigma (σ), which is the distance at which the potential reaches its minimum, and epsilon (ε), which is the depth of the potential well. 
# The Lennard-Jones potential is given by the formula: 
# E(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between the particles. 
# The term (σ/r)^12 represents the repulsive forces due to overlapping electron orbitals, while (σ/r)^6 represents the attractive van der Waals forces. 
# The potential approaches zero as the distance r becomes much larger than σ, indicating negligible interaction at large separations.

def E_ij(r, sigma, epsilon):
    '''Calculate the Lennard-Jones potential energy between two particles.
    Parameters:
    r : float
        The distance between the two particles.
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The potential energy between the two particles at distance r.
    '''
    if r <= 0:
        raise ValueError("Distance r must be positive and non-zero.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive and non-zero.")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative.")

    # Calculate the ratio of sigma to r
    sr_ratio = sigma / r
    
    # Calculate the Lennard-Jones potential using the formula
    E_lj = 4 * epsilon * (sr_ratio**12 - sr_ratio**6)
    
    return E_lj



# Background: In molecular dynamics simulations, calculating the total energy of a single particle 
# involves summing up the interaction energies between that particle and all other particles in the system. 
# The Lennard-Jones potential, which models the interaction between a pair of particles, is used for this purpose. 
# Given the periodic boundary conditions, the minimum image distance is used to ensure that the shortest 
# possible distance is considered for each pairwise interaction. The total energy of a particle is the sum 
# of the Lennard-Jones potential energies with all other particles, considering these minimum image distances.


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
    def dist(r1, r2, L):
        '''Calculate the minimum image distance between two atoms in a periodic cubic system.'''
        delta = r2 - r1
        delta = delta - L * np.round(delta / L)
        return np.sqrt(np.sum(delta**2))

    def E_ij(r, sigma, epsilon):
        '''Calculate the Lennard-Jones potential energy between two particles.'''
        sr_ratio = sigma / r
        return 4 * epsilon * (sr_ratio**12 - sr_ratio**6)

    total_energy = 0.0
    for pos in positions:
        if not np.array_equal(r, pos):  # Ensure we do not calculate self-interaction
            distance = dist(r, pos, L)
            energy = E_ij(distance, sigma, epsilon)
            total_energy += energy

    return total_energy

from scicode.parse.parse import process_hdf5_to_tuple
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
