import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro



# Background: In simulations involving particles within a defined space, it's common to use periodic boundary conditions (PBCs) to simulate an infinite system. 
# This technique involves wrapping the coordinates of a particle such that if a particle moves out of one side of a cubic simulation box, 
# it reappears on the opposite side. Mathematically, this is achieved by using modular arithmetic. 
# For a given coordinate, the wrapped position is calculated as the remainder of the division of the coordinate by the box length L, 
# adjusted to ensure the position remains within the bounds of 0 and L.

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Convert the input to a numpy array if it isn't already, for ease of computation.
    r = np.array(r)
    
    # Apply periodic boundary conditions using the modulo operation
    coord = r % L
    
    return coord

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('77.1', 3)
target = targets[0]

particle_position = np.array([10.5, -1.2, 20.3])
box_length = 5.0
# Applying the wrap function
assert np.allclose(wrap(particle_position, box_length), target)
target = targets[1]

particle_position1 = np.array([10.0, 5.5, -0.1])
box_length1 = 10.0
# Applying the wrap function
assert np.allclose(wrap(particle_position1, box_length1), target)
target = targets[2]

particle_position2 = np.array([23.7, -22.1, 14.3])
box_length2 = 10.0
# Applying the wrap function
assert np.allclose(wrap(particle_position2, box_length2), target)
