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
    # Check if L is zero or negative to avoid division by zero and logical errors
    if L <= 0:
        raise ValueError("Box size L must be positive and non-zero.")
    
    # Convert input to numpy array if it's not already one
    r = np.asarray(r)
    
    # Check if the input contains non-numeric values
    if not np.issubdtype(r.dtype, np.number):
        raise ValueError("Input coordinates must be numeric.")
    
    # Check for infinite or NaN values in the coordinates
    if np.any(np.isinf(r)) or np.any(np.isnan(r)):
        raise ValueError("Coordinates must be finite numbers.")
    
    # Use numpy's modulus operation to wrap the coordinates
    coord = np.mod(r, L)
    return coord



# Background: In molecular dynamics simulations, the minimum image convention is used to calculate the shortest distance between two particles in a periodic system. This is crucial for accurately computing interactions in a system with periodic boundary conditions. The minimum image distance is the smallest distance between two particles, considering that each particle can be represented by an infinite number of periodic images. For a cubic box of size L, the minimum image distance can be calculated by considering the displacement vector between two particles and adjusting it to ensure it is the shortest possible vector within the periodic boundaries.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Convert inputs to numpy arrays
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    
    # Calculate the displacement vector
    delta = r2 - r1
    
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
