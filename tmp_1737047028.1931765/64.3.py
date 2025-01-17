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


from scicode.parse.parse import process_hdf5_to_tuple

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
