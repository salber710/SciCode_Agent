import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro

# Background: In computational simulations, especially those involving molecular dynamics or particle systems, 
# it is common to use periodic boundary conditions (PBC) to simulate an infinite system using a finite-sized 
# simulation box. This approach helps to minimize edge effects and allows particles to move seamlessly across 
# the boundaries of the simulation box. When a particle exits one side of the box, it re-enters from the 
# opposite side. Mathematically, this is achieved by wrapping the particle's coordinates back into the box 
# using the modulo operation. For a cubic box of size L, if a particle's coordinate in any dimension exceeds 
# L or is less than 0, it is wrapped back into the range [0, L) using the formula: 
# wrapped_coordinate = coordinate % L. This ensures that all particle coordinates remain within the bounds 
# of the simulation box.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    coord = np.mod(r, L)
    return coord



# Background: In molecular dynamics simulations, the minimum image convention is used to calculate the shortest
# distance between two particles in a periodic system. This is crucial for accurately computing interactions
# between particles in a system with periodic boundary conditions. The minimum image distance ensures that
# the distance calculation considers the closest image of a particle, accounting for the periodicity of the
# simulation box. For a cubic box of size L, the minimum image distance between two particles with coordinates
# r1 and r2 is calculated by considering the displacement vector between them and adjusting it to lie within
# the range [-L/2, L/2) in each dimension. This adjustment is done by subtracting L from the displacement if
# it is greater than L/2, or adding L if it is less than -L/2. The Euclidean distance is then computed using
# this adjusted displacement vector.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the displacement vector
    delta = np.array(r2) - np.array(r1)
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(delta)
    
    return distance


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('77.2', 3)
target = targets[0]

r1 = np.array([2.0, 3.0, 4.0])
r2 = np.array([2.5, 3.5, 4.5])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
target = targets[1]

r1 = np.array([1.0, 1.0, 1.0])
r2 = np.array([9.0, 9.0, 9.0])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
target = targets[2]

r1 = np.array([0.1, 0.1, 0.1])
r2 = np.array([9.9, 9.9, 9.9])
box_length = 10.0
assert np.allclose(dist(r1, r2, box_length), target)
