import numpy as np
import itertools

# Background: In computational simulations, especially those involving particles in a confined space, it is common to use periodic boundary conditions (PBCs). 
# PBCs are used to simulate an infinite system by wrapping particles that move out of one side of the simulation box back into the opposite side. 
# This is particularly useful in molecular dynamics simulations to avoid edge effects and to mimic a bulk environment. 
# The idea is to ensure that any coordinate that exceeds the boundaries of the box is wrapped back into the box. 
# For a cubic box of size L, if a coordinate x is less than 0, it should be wrapped to x + L, and if it is greater than or equal to L, it should be wrapped to x - L. 
# This can be efficiently achieved using the modulo operation.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use numpy's modulo operation to wrap the coordinates
    coord = np.mod(r, L)
    return coord


# Background: In molecular simulations with periodic boundary conditions, the minimum image convention is used to calculate the shortest distance between two particles. 
# This is crucial for accurately computing interactions in a periodic system. The minimum image distance is the smallest distance between two particles, considering that 
# each particle can be represented by an infinite number of periodic images. For a cubic box of size L, the minimum image distance between two coordinates r1 and r2 
# is calculated by considering the direct distance and the distances obtained by shifting one of the coordinates by ±L in each dimension. 
# The minimum image distance ensures that the calculated distance is the shortest possible, which is essential for correctly modeling interactions in a periodic system.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the difference vector between the two positions
    delta = np.array(r1) - np.array(r2)
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    # Calculate the Euclidean distance
    distance = np.sqrt(np.sum(delta**2))
    
    return distance


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model van der Waals forces. The potential is characterized by two parameters: 
# sigma (σ), which is the finite distance at which the inter-particle potential is zero, and epsilon (ε), which is the depth of the potential well. 
# The Lennard-Jones potential is given by the formula:
# E(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6]
# where r is the distance between the particles. The term (σ/r)^12 represents the repulsive forces, which dominate at short ranges, 
# while the term (σ/r)^6 represents the attractive forces, which dominate at longer ranges. The potential reaches its minimum value of -ε at r = σ.

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
    # Calculate the ratio of sigma to r
    sr_ratio = sigma / r
    
    # Calculate the Lennard-Jones potential using the formula
    E_lj = 4 * epsilon * (sr_ratio**12 - sr_ratio**6)
    
    return E_lj


# Background: In molecular dynamics simulations, calculating the total energy of a single particle involves summing up the interaction energies between that particle and all other particles in the system. 
# The Lennard-Jones potential is often used to model these interactions. Given the periodic nature of the system, the minimum image convention is used to calculate the shortest distance between particles, 
# ensuring that the interactions are computed correctly across periodic boundaries. The function E_ij provides the potential energy between two particles based on their distance, 
# and this function will be used to sum the energies for a single particle with all others, considering periodic boundary conditions.


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
    total_energy = 0.0
    for pos in positions:
        if not np.array_equal(r, pos):  # Ensure we do not calculate self-interaction
            # Calculate the minimum image distance
            delta = np.array(r) - np.array(pos)
            delta = delta - L * np.round(delta / L)
            distance = np.sqrt(np.sum(delta**2))
            
            # Calculate the Lennard-Jones potential energy
            energy = E_ij(distance, sigma, epsilon)
            total_energy += energy
    
    return total_energy



# Background: In molecular dynamics simulations, the total energy of a system is the sum of all pairwise interaction energies between particles. 
# The Lennard-Jones potential is commonly used to model these interactions. To calculate the total energy of the system, we need to consider 
# the interaction between every unique pair of particles, ensuring that each pair is only counted once. This is typically done using a double 
# loop over all particles, but only considering pairs where the first particle index is less than the second. The minimum image convention is 
# used to calculate the shortest distance between particles in a periodic system, ensuring accurate energy calculations across periodic boundaries.



def E_system(positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of the system.
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
    total_E = 0.0
    num_particles = len(positions)
    
    # Iterate over all unique pairs of particles
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            # Calculate the minimum image distance
            delta = np.array(positions[i]) - np.array(positions[j])
            delta = delta - L * np.round(delta / L)
            distance = np.sqrt(np.sum(delta**2))
            
            # Calculate the Lennard-Jones potential energy for this pair
            energy = E_ij(distance, sigma, epsilon)
            total_E += energy
    
    return total_E


from scicode.parse.parse import process_hdf5_to_tuple

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
