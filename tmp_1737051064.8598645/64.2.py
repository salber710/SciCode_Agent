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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('64.2', 3)
target = targets[0]

r1 = np.array([2.0, 3.0, 4.0])
r2 = np.array([2.5, 3.5, 4.5])
box_length = 10.0
distance1 = dist(r1, r2, box_length)
assert np.allclose(distance1[0], target)  # Expected distance should be roughly 0.866
target = targets[1]

r1 = np.array([1.0, 1.0, 1.0])
r2 = np.array([9.0, 9.0, 9.0])
box_length = 10.0
distance2 = dist(r1, r2, box_length)
assert np.allclose(distance2[0], target)  # Expected distance should be sqrt(12)
target = targets[2]

r1 = np.array([0.1, 0.1, 0.1])
r2 = np.array([9.9, 9.9, 9.9])
box_length = 10.0
distance3 = dist(r1, r2, box_length)
assert np.allclose(distance3[0], target)  # Expected distance should be roughly sqrt(0.12)
