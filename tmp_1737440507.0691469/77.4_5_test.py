import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro

# Background: In molecular simulations, periodic boundary conditions (PBCs) are often used to model a small part of a larger system without edge effects. 
# This involves wrapping particles back into the simulation box when they move out, creating an effect of infinite tiling of the box.
# When a particle's coordinate exceeds the box's boundary, it re-enters from the opposite side, ensuring continuity.
# Mathematically, if a coordinate component of a particle is outside the range [0, L), it is wrapped back using modulo operation.
# This helps maintain the particles within the defined simulation space, which is crucial for accurate simulation results.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use the modulo operation to wrap the coordinates
    coord = np.mod(r, L)
    return coord


# Background: In molecular dynamics simulations with periodic boundary conditions, the minimum image convention is used to calculate the shortest distance between two particles in a periodic system. 
# This is crucial for accurately determining interactions, such as forces and potential energy, between particles that may appear on opposite sides of the simulation box.
# The minimum image distance is determined by considering the direct distance between two particles along with distances across periodic boundaries, ensuring that the shortest path through the periodic images is chosen.
# Mathematically, this involves first computing the displacement vector between the particles, then adjusting this vector by considering the periodicity of the box. 
# The displacement vector component for each dimension is adjusted by subtracting the nearest multiple of the box length if the distance exceeds half the box length.
# This ensures that the distance calculation reflects the nearest image of each particle.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    # Calculate the displacement vector between the two atoms
    delta = np.array(r2) - np.array(r1)
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    # Calculate and return the Euclidean distance
    distance = np.linalg.norm(delta)
    return distance


# Background: In molecular simulations using periodic boundary conditions, it is important to calculate not just the minimum image distance but also the minimum image vector between two particles. 
# The minimum image vector is the displacement vector between two particles considering the periodic boundaries, ensuring that it points to the closest image of the second particle relative to the first.
# This vector is crucial for determining the direction of forces between particles, which is essential for simulations that calculate dynamics or interactions.
# Like the minimum image distance, this involves computing the displacement vector and adjusting each component by subtracting the nearest multiple of the box length if the displacement exceeds half the box length.
# This adjustment ensures the vector points to the nearest periodic image, providing accurate directional information.

def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    '''
    # Calculate the displacement vector between the two atoms
    delta = np.array(r2) - np.array(r1)
    
    # Apply the minimum image convention to get the minimum image vector
    delta = delta - L * np.round(delta / L)
    
    return delta



# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules.
# It is widely used in molecular simulations to model van der Waals forces. The potential function is characterized by two parameters:
# sigma (σ), which is the distance at which the potential is zero, and epsilon (ε), which represents the depth of the potential well.
# The Lennard-Jones potential V(r) is given by V(r) = 4ε[(σ/r)^12 - (σ/r)^6], where r is the distance between the particles.
# In simulations, it is common to truncate and shift the potential at a cutoff distance rc to improve computational efficiency.
# The potential is set to zero for distances greater than rc, and the potential is shifted such that V(rc) = 0, ensuring no discontinuity at the cutoff.

def E_ij(r, sigma, epsilon, rc):
    '''Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The combined potential energy between the two particles, considering the specified potentials.
    '''
    
    if r < rc:
        # Calculate the Lennard-Jones potential
        lj_potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        
        # Calculate the Lennard-Jones potential at the cutoff
        lj_cutoff = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
        
        # Shift the potential so that it is zero at the cutoff
        E = lj_potential - lj_cutoff
    else:
        # If the distance is greater than or equal to the cutoff, the potential is zero
        E = 0.0
        
    return E

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('77.4', 3)
target = targets[0]

r1 = 1.0  # Close to the sigma value
sigma1 = 1.0
epsilon1 = 1.0
rc = 1
assert np.allclose(E_ij(r1, sigma1, epsilon1, rc), target)
target = targets[1]

r2 = 0.5  # Significantly closer than the effective diameter
sigma2 = 1.0
epsilon2 = 1.0
rc = 2
assert np.allclose(E_ij(r2, sigma2, epsilon2, rc), target)
target = targets[2]

r3 = 2.0  # Larger than sigma
sigma3 = 1.0
epsilon3 = 1.0
rc = 3
assert np.allclose(E_ij(r3, sigma3, epsilon3, rc), target)
