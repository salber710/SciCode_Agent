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


# Background: In molecular dynamics simulations, the minimum image vector is used to determine the shortest
# vector between two particles in a periodic system. This is essential for calculating forces and interactions
# accurately in a system with periodic boundary conditions. The minimum image vector ensures that the vector
# calculation considers the closest image of a particle, taking into account the periodicity of the simulation
# box. For a cubic box of size L, the minimum image vector between two particles with coordinates r1 and r2
# is calculated by determining the displacement vector between them and adjusting it to lie within the range
# [-L/2, L/2) in each dimension. This adjustment is done by subtracting L from the displacement if it is greater
# than L/2, or adding L if it is less than -L/2. The resulting vector is the minimum image vector, which can
# then be used for further calculations such as force computations.

def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    '''
    # Calculate the displacement vector
    delta = np.array(r2) - np.array(r1)
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    return delta



# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model van der Waals forces. The potential is characterized by two parameters: 
# sigma (σ), which is the distance at which the potential is zero, and epsilon (ε), which is the depth of the potential well. 
# The Lennard-Jones potential is given by the formula: 
# V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between the particles. 
# To avoid computational inefficiency, the potential is often truncated and shifted to zero at a cutoff distance rc. 
# This means that for distances greater than rc, the potential is considered to be zero. 
# The truncated and shifted potential is given by: 
# V_shifted(r) = V(r) - V(rc) for r < rc, and V_shifted(r) = 0 for r >= rc.

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
        # Calculate the Lennard-Jones potential at distance r
        V_r = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        # Calculate the Lennard-Jones potential at the cutoff distance rc
        V_rc = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
        # Return the truncated and shifted potential
        return V_r - V_rc
    else:
        # If the distance is greater than or equal to the cutoff, the potential is zero
        return 0.0


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
