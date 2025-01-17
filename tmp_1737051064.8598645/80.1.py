import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro



# Background: In computational chemistry and molecular dynamics simulations, the concept of minimum image distance is crucial for calculating distances between particles in a periodic system. 
# A periodic cubic system is a model where the simulation box is repeated infinitely in all directions. This is used to mimic bulk properties of materials without edge effects.
# The minimum image convention is used to calculate the shortest distance between two particles, considering the periodic boundaries. 
# For a cubic box of side length L, the minimum image distance between two points r1 and r2 is calculated by considering the closest image of r2 to r1, 
# which can be found by adjusting the coordinates of r2 by multiples of L to find the minimum distance.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Convert inputs to numpy arrays for vectorized operations
    r1 = np.array(r1)
    r2 = np.array(r2)
    
    # Calculate the difference vector
    delta = r2 - r1
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    # Calculate the Euclidean distance
    distance = np.linalg.norm(delta)
    
    return distance


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('80.1', 3)
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
