import numpy as np



# Background: In computational simulations, especially those involving particles in a confined space, 
# it is common to use periodic boundary conditions (PBCs) to simulate an infinite system using a finite 
# simulation box. When a particle moves out of one side of the box, it re-enters from the opposite side. 
# This is akin to the concept of wrapping around in a toroidal space. The goal of applying PBCs is to 
# ensure that the coordinates of particles remain within the bounds of the simulation box. For a cubic 
# box of size L, if a coordinate exceeds L, it should be wrapped around by subtracting L, and if it is 
# less than 0, it should be wrapped by adding L. This ensures that all coordinates are within the range 
# [0, L).


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

targets = process_hdf5_to_tuple('60.1', 3)
target = targets[0]

particle_position = np.array([10.5, -1.2, 20.3])
box_length = 10.0
# Applying the wrap function
assert np.allclose(wrap(particle_position, box_length), target) # Expected output: [0.5, 8.8, 0.3]
target = targets[1]

particle_position1 = np.array([10.0, 5.5, -0.1])
box_length1 = 10.0
# Applying the wrap function
assert np.allclose(wrap(particle_position1, box_length1), target)  # Expected output: [0.0, 5.5, 9.9]
target = targets[2]

particle_position2 = np.array([23.7, -22.1, 14.3])
box_length2 = 10.0
# Applying the wrap function
assert np.allclose(wrap(particle_position2, box_length2), target)  # Expected output: [3.7, 7.9, 4.3]
