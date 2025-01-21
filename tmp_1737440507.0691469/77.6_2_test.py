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
# It is a function of the distance between the particles and is characterized by two parameters: sigma (σ) and epsilon (ε).
# Sigma (σ) is the distance at which the potential is zero, and epsilon (ε) is the depth of the potential well, representing the strength of attraction.
# The Lennard-Jones potential is given by the formula: V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between the particles.
# To avoid infinite interactions in simulations, the potential is often truncated and shifted. The truncation is done at a specified cutoff distance (rc),
# beyond which the potential is not considered, and the potential is shifted to ensure continuity at the cutoff.
# The shifted potential ensures that V(rc) = 0, which helps in avoiding discontinuities in simulations.
# The shifted potential can be calculated by subtracting the potential value at the cutoff distance from the potential.

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
    # Compute the Lennard-Jones potential at distance r
    if r < rc:
        # Calculate (sigma / r)^6
        sr6 = (sigma / r) ** 6
        # Lennard-Jones potential
        V_r = 4 * epsilon * (sr6**2 - sr6)
        
        # Calculate the potential at the cutoff distance rc
        sr6_cutoff = (sigma / rc) ** 6
        V_rc = 4 * epsilon * (sr6_cutoff**2 - sr6_cutoff)
        
        # Calculate the shifted potential
        E = V_r - V_rc
    else:
        # Beyond rc, the potential is zero
        E = 0.0

    return E


# Background: The force between two particles interacting through the Lennard-Jones potential can be derived
# from the potential energy function by taking its negative gradient with respect to the distance between the particles.
# The Lennard-Jones force is an attractive force at long distances and a repulsive force at short distances, reflecting
# the attraction and repulsion terms in the potential. The force is given by the derivative of the Lennard-Jones potential:
# F(r) = -dV(r)/dr = 24 * ε * [(2 * (σ/r)^12) - ((σ/r)^6)] * (1/r), where V(r) is the Lennard-Jones potential.
# Like the potential, the force must be truncated and shifted to zero beyond the cutoff distance for computational efficiency
# in simulations. The force vector is aligned with the displacement vector between the particles, scaled by the magnitude
# of the force.

def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    # If the distance is less than the cutoff, calculate the force
    if r < rc:
        # Calculate (sigma / r)^6
        sr6 = (sigma / r) ** 6
        # Calculate the magnitude of the Lennard-Jones force
        force_magnitude = 24 * epsilon * (2 * sr6**2 - sr6) / r
        
        # The force vector is directed along the displacement vector
        # Since only the magnitude is returned, in practical applications, this will be multiplied by the
        # normalized displacement vector to get the force vector components in each dimension.
        return force_magnitude
    else:
        # Beyond rc, the force is zero
        return 0.0



# Background: In molecular dynamics simulations, while using the Lennard-Jones potential, the potential is often truncated at a cutoff distance to save computational resources. 
# However, this truncation can lead to an underestimation of the total potential energy of the system because it ignores the interactions beyond this cutoff. 
# To correct for this truncation, a tail correction is applied to the energy. 
# This correction estimates the contribution to the potential energy from the interactions that are not explicitly calculated.
# The tail correction for the energy in a cubic system is derived by integrating the Lennard-Jones potential beyond the cutoff distance to infinity.
# The formula for the tail correction of the energy is: E_tail = (8/3) * π * N * (N-1) * ρ * ε * [ (1/3) * (σ/rc)^9 - (σ/rc)^3 ]
# where N is the number of particles, ρ is the number density, ε is the depth of the potential well, σ is the distance at which the inter-particle potential is zero, 
# and rc is the cutoff distance. The correction accounts for the potential energy from particles interacting beyond rc.

def E_tail(N, L, sigma, epsilon, rc):
    '''Calculate the energy tail correction for a system of particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Length of cubic box.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    '''

    # Calculate the number density
    rho = N / (L**3)
    
    # Calculate (sigma / rc)^3 and (sigma / rc)^9
    sr3 = (sigma / rc) ** 3
    sr9 = sr3 ** 3
    
    # Calculate the energy tail correction
    E_tail_LJ = (8 / 3) * math.pi * N * (N - 1) * rho * epsilon * ((1 / 3) * sr9 - sr3)
    
    return E_tail_LJ

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('77.6', 3)
target = targets[0]

N=2
L=10
sigma = 1
epsilon = 1
rc = 1
assert np.allclose(E_tail(N,L,sigma,epsilon,rc), target)
target = targets[1]

N=5
L=10
sigma = 1
epsilon = 1
rc = 5
assert np.allclose(E_tail(N,L,sigma,epsilon,rc), target)
target = targets[2]

N=10
L=10
sigma = 1
epsilon = 1
rc = 9
assert np.allclose(E_tail(N,L,sigma,epsilon,rc), target)
