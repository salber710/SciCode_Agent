import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro



# Background: In computational simulations, especially those involving molecular dynamics or particle systems, it is common to use periodic boundary conditions (PBCs) to simulate an infinite system using a finite-sized simulation box. This approach helps to minimize edge effects and allows particles to move seamlessly across the boundaries of the simulation box. When a particle moves out of one side of the box, it re-enters from the opposite side. The process of applying PBCs involves wrapping the coordinates of particles such that they remain within the bounds of the simulation box. For a cubic box of size L, this can be achieved by taking the modulus of the particle's coordinates with respect to L.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use numpy's modulus operation to wrap the coordinates
    coord = np.mod(r, L)
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
