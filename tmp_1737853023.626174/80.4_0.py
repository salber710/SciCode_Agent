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
    if abs(dx) > L / 2:
        dx -= L * round(dx / L)
    if abs(dy) > L / 2:
        dy -= L * round(dy / L)
    if abs(dz) > L / 2:
        dz -= L * round(dz / L)
    
    # Calculate the Euclidean distance using the adjusted coordinates
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    
    return distance


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model van der Waals forces. The potential is characterized by two parameters: 
# sigma (σ), which is the distance at which the potential is zero, and epsilon (ε), which is the depth of the potential well. 
# The Lennard-Jones potential is given by the formula: 
# V_LJ(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between the particles. 
# To avoid infinite interactions in simulations, the potential is often truncated and shifted to zero at a cutoff distance rc. 
# This means that for distances greater than rc, the potential is set to zero, and for distances less than rc, the potential is adjusted 
# so that it smoothly goes to zero at rc. This is done by subtracting the potential value at rc from the potential at r.

def E_ij(r, sigma, epsilon, rc):
    '''Calculate the truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The potential energy between the two particles, considering the Lennard-Jones potential.
    '''
    if r <= 0:
        raise ValueError("Distance r must be positive and non-zero.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive to avoid division by zero.")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative as it represents the depth of the potential well.")
    if rc < 0:
        raise ValueError("Cutoff distance rc must be non-negative.")
    
    if r >= rc:
        # If the distance is greater than or equal to the cutoff, the potential is zero
        E = 0.0
    else:
        # Calculate the Lennard-Jones potential at distance r
        lj_potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        # Calculate the Lennard-Jones potential at the cutoff distance rc
        lj_potential_rc = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
        # Truncate and shift the potential
        E = lj_potential - lj_potential_rc
    
    return E


# Background: In molecular dynamics simulations, the total potential energy of a system is a crucial quantity that
# represents the sum of all pairwise interactions between particles. For systems modeled using the Lennard-Jones
# potential, the total energy is computed by summing the potential energy contributions from all unique pairs of
# particles. The Lennard-Jones potential accounts for both attractive and repulsive forces between particles, and
# is characterized by parameters sigma (σ) and epsilon (ε). The potential is truncated and shifted to zero at a
# cutoff distance rc to ensure computational efficiency and to avoid infinite interactions. The minimum image
# convention is used to calculate the shortest distance between particles in a periodic cubic system, ensuring
# that the periodic boundary conditions are respected.


def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Length of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The total potential energy of the system (in zeptojoules).
    '''
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")
    
    if epsilon <= 0:
        raise ValueError("Epsilon must be positive.")
    
    if rc <= 0:
        raise ValueError("Cutoff distance rc must be positive.")
    
    if not isinstance(xyz, np.ndarray):
        raise TypeError("xyz must be a NumPy array.")
    
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be a two-dimensional array with three columns.")

    def dist(r1, r2, L):
        '''Calculate the minimum image distance between two atoms in a periodic cubic system.'''
        dx = r2[0] - r1[0]
        dy = r2[1] - r1[1]
        dz = r2[2] - r1[2]
        
        dx -= L * round(dx / L)
        dy -= L * round(dy / L)
        dz -= L * round(dz / L)
        
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def E_ij(r, sigma, epsilon, rc):
        '''Calculate the truncated and shifted Lennard-Jones potential energy between two particles.'''
        if r >= rc:
            return 0.0
        elif r == 0:
            return float('inf')  # Handle division by zero as infinite potential energy
        else:
            lj_potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
            lj_potential_rc = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
            return lj_potential - lj_potential_rc

    N = xyz.shape[0]
    total_energy = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r = dist(xyz[i], xyz[j], L)
            total_energy += E_ij(r, sigma, epsilon, rc)

    return total_energy



# Background: The Lennard-Jones force is derived from the Lennard-Jones potential, which models the interaction
# between a pair of neutral atoms or molecules. The force is the negative gradient of the potential with respect
# to the distance between the particles. For the Lennard-Jones potential, the force can be expressed as:
# F_LJ(r) = -dV_LJ/dr = 24 * ε * [(2 * (σ/r)^12) - ((σ/r)^6)] / r, where r is the distance between the particles.
# This force accounts for both repulsive and attractive interactions. The force is truncated to zero at a cutoff
# distance rc, similar to the potential, to ensure computational efficiency. The force vector is directed along
# the line connecting the two particles, and its magnitude is given by the expression above.

def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering both the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The magnitude of the force experienced by particle i due to particle j, considering the specified potentials.
    '''
    if r >= rc:
        # If the distance is greater than or equal to the cutoff, the force is zero
        return 0.0
    elif r == 0:
        # Handle division by zero as infinite force
        return float('inf')
    else:
        # Calculate the magnitude of the Lennard-Jones force
        force_magnitude = 24 * epsilon * ((2 * (sigma / r)**12) - ((sigma / r)**6)) / r
        return force_magnitude

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('80.4', 3)
target = targets[0]

sigma = 1
epsilon = 1
r = np.array([-3.22883506e-03,  2.57056485e+00,  1.40822287e-04])
rc = 1
assert np.allclose(f_ij(r,sigma,epsilon,rc), target)
target = targets[1]

sigma = 2
epsilon = 3
r = np.array([3,  -4,  5])
rc = 10
assert np.allclose(f_ij(r,sigma,epsilon,rc), target)
target = targets[2]

sigma = 3
epsilon = 7
r = np.array([5,  9,  7])
rc = 20
assert np.allclose(f_ij(r,sigma,epsilon,rc), target)
