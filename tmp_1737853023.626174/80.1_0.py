import os
import math
import time
import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import pickle
from scipy.constants import  Avogadro



# Background: In computational chemistry and physics, when dealing with periodic boundary conditions in a cubic system,
# it is important to calculate the minimum image distance between two points (atoms) in the system. This is because
# the system is periodic, meaning that it repeats itself in all directions. The minimum image convention is used to
# find the shortest distance between two points, taking into account the periodicity of the system. The idea is to
# consider the closest image of the second point to the first point, which may be in the original box or in one of
# the neighboring periodic images. The minimum image distance is calculated by considering the distance in each
# dimension and adjusting it if the distance is greater than half the box length, L/2, by subtracting L. This ensures
# that the shortest path is always considered.

def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the difference in each dimension
    dx = r2[0] - r1[0]
    dy = r2[1] - r1[1]
    dz = r2[2] - r1[2]
    
    # Apply the minimum image convention
    dx = dx - L * round(dx / L)
    dy = dy - L * round(dy / L)
    dz = dz - L * round(dz / L)
    
    # Calculate the Euclidean distance using the adjusted coordinates
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
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
