
# Background: Periodic boundary conditions are commonly used in simulations of materials and molecular systems to 
# mimic an infinite system using a finite-sized simulation box. When a particle moves out of one side of the 
# simulation box, it re-enters from the opposite side. This is akin to wrapping the coordinates of the particle 
# such that they remain within the boundaries of the box. For a cubic box of length L, coordinates can be wrapped 
# using the modulus operation. This ensures that any coordinate value that exceeds the box dimensions is translated 
# back within the box, maintaining a continuous system without edge effects.

import numpy as np

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    coord = r % L
    return coord
