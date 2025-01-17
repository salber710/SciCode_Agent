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


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model van der Waals forces. The potential is characterized by two parameters: 
# sigma (σ), which is the distance at which the potential is zero, and epsilon (ε), which is the depth of the potential well. 
# The Lennard-Jones potential is given by the formula: 
# V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between the particles. 
# To avoid infinite interactions in simulations, the potential is often truncated and shifted to zero at a cutoff distance rc. 
# This means that for r > rc, the potential energy is set to zero. The truncated and shifted potential is given by:
# V_shifted(r) = V(r) - V(rc) for r <= rc, and V_shifted(r) = 0 for r > rc.

def E_ij(r, sigma, epsilon, rc):
    '''Calculate the truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The potential energy between the two particles, considering the specified potentials.
    '''
    if r > rc:
        return 0.0
    else:
        # Calculate the Lennard-Jones potential
        lj_potential = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
        # Calculate the Lennard-Jones potential at the cutoff distance
        lj_potential_rc = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)
        # Truncate and shift the potential
        E = lj_potential - lj_potential_rc
        return E


# Background: In molecular dynamics simulations, the total potential energy of a system is a crucial quantity that helps in understanding the stability and behavior of the system. 
# The total potential energy is calculated by summing up the pairwise potential energies between all particles in the system. 
# For a system of N particles, this involves calculating the potential energy for each unique pair of particles (i, j) where i < j, using the Lennard-Jones potential. 
# The Lennard-Jones potential is truncated and shifted to zero at a cutoff distance rc to avoid infinite interactions. 
# The minimum image convention is used to calculate the shortest distance between particles in a periodic cubic system, ensuring that the periodic boundaries are correctly handled.


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
    def dist(r1, r2, L):
        '''Calculate the minimum image distance between two atoms in a periodic cubic system.'''
        delta = r2 - r1
        delta = delta - L * np.round(delta / L)
        return np.linalg.norm(delta)

    def E_ij(r, sigma, epsilon, rc):
        '''Calculate the truncated and shifted Lennard-Jones potential energy between two particles.'''
        if r > rc:
            return 0.0
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



# Background: In molecular dynamics, the force between two particles can be derived from the potential energy function. 
# For the Lennard-Jones potential, the force is the negative gradient of the potential with respect to the distance between particles. 
# The Lennard-Jones force is given by the derivative of the potential: 
# F(r) = -dV/dr = 24 * epsilon * [(2 * (sigma/r)^12) - ((sigma/r)^6)] / r. 
# This force acts along the line joining the two particles and is attractive at long ranges and repulsive at short ranges. 
# When the potential is truncated and shifted at a cutoff distance rc, the force is set to zero for r > rc.

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
    if r > rc:
        return 0.0
    else:
        # Calculate the Lennard-Jones force
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
