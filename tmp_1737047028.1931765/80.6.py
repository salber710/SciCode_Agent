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


# Background: In molecular dynamics simulations, calculating the forces on each particle due to interactions with other particles is crucial for understanding the system's dynamics. 
# The force on a particle is derived from the potential energy function, specifically the Lennard-Jones potential in this context. 
# The force between two particles is calculated using the negative gradient of the potential energy with respect to the distance between them. 
# For the Lennard-Jones potential, the force is given by: 
# F(r) = -dV/dr = 24 * epsilon * [(2 * (sigma/r)^12) - ((sigma/r)^6)] / r. 
# This force is computed for each pair of particles within a cutoff distance rc, and the net force on each particle is the vector sum of all pairwise forces acting on it. 
# The minimum image convention is used to account for periodic boundary conditions, ensuring that the shortest distance is considered.


def forces(N, xyz, L, sigma, epsilon, rc):
    '''Calculate the net forces acting on each particle in a system due to all pairwise interactions.
    Parameters:
    N : int
        The number of particles in the system.
    xyz : ndarray
        A NumPy array with shape (N, 3) containing the positions of each particle in the system,
        in nanometers.
    L : float
        The length of the side of the cubic simulation box (in nanometers), used for applying the minimum
        image convention in periodic boundary conditions.
    sigma : float
        The Lennard-Jones size parameter (in nanometers), indicating the distance at which the
        inter-particle potential is zero.
    epsilon : float
        The depth of the potential well (in zeptojoules), indicating the strength of the particle interactions.
    rc : float
        The cutoff distance (in nanometers) beyond which the inter-particle forces are considered negligible.
    Returns:
    ndarray
        A NumPy array of shape (N, 3) containing the net force vectors acting on each particle in the system,
        in zeptojoules per nanometer (zJ/nm).
    '''
    # Initialize the force array
    f_xyz = np.zeros((N, 3))

    # Function to calculate the minimum image distance
    def dist_vector(r1, r2, L):
        delta = r2 - r1
        delta = delta - L * np.round(delta / L)
        return delta

    # Function to calculate the Lennard-Jones force
    def f_ij(r_vec, r, sigma, epsilon, rc):
        if r > rc:
            return np.zeros(3)
        else:
            force_magnitude = 24 * epsilon * ((2 * (sigma / r)**12) - ((sigma / r)**6)) / r
            return force_magnitude * (r_vec / r)

    # Calculate forces
    for i in range(N):
        for j in range(i + 1, N):
            r_vec = dist_vector(xyz[i], xyz[j], L)
            r = np.linalg.norm(r_vec)
            force = f_ij(r_vec, r, sigma, epsilon, rc)
            f_xyz[i] += force
            f_xyz[j] -= force  # Newton's third law

    return f_xyz



# Background: The Velocity Verlet algorithm is a numerical method used to integrate Newton's equations of motion. 
# It is particularly useful in molecular dynamics simulations for updating the positions and velocities of particles 
# over time. The algorithm is favored for its simplicity and stability. The basic steps of the Velocity Verlet algorithm 
# are as follows:
# 1. Update positions using current velocities and accelerations.
# 2. Calculate the new forces based on the updated positions.
# 3. Update velocities using the average of the old and new accelerations.
# In the context of a system of particles interacting through the Lennard-Jones potential, the forces are calculated 
# using the potential's gradient. The minimum image convention is used to handle periodic boundary conditions, ensuring 
# that the shortest distance is considered. The algorithm requires the current positions, velocities, time step size, 
# and particle mass to compute the new positions and velocities.

def velocity_verlet(N, sigma, epsilon, positions, velocities, dt, m, L, rc):
    '''This function runs Velocity Verlet algorithm to integrate the positions and velocities of atoms interacting through
    Lennard Jones Potential forward for one time step according to Newton's Second Law.
    Inputs:
    N: int
       The number of particles in the system.
    sigma: float
       the distance at which Lennard Jones potential reaches zero
    epsilon: float
             potential well depth of Lennard Jones potential
    positions: 2D array of floats with shape (N,3)
              current positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    velocities: 2D array of floats with shape (N,3)
              current velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    dt: float
        time step size
    m: float
       mass
    Outputs:
    new_positions: 2D array of floats with shape (N,3)
                   new positions of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    new_velocities: 2D array of floats with shape (N,3)
              current velocities of all atoms, N is the number of atoms, 3 is x,y,z coordinate
    '''

    # Calculate initial forces
    forces_initial = forces(N, positions, L, sigma, epsilon, rc)

    # Update positions
    new_positions = positions + velocities * dt + 0.5 * (forces_initial / m) * dt**2

    # Calculate new forces based on updated positions
    forces_new = forces(N, new_positions, L, sigma, epsilon, rc)

    # Update velocities
    new_velocities = velocities + 0.5 * (forces_initial + forces_new) / m * dt

    return new_positions, new_velocities


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('80.6', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
N = 2
positions =np.array([[5.53189438, 7.24107158, 6.12400066],
 [5.48717536, 4.31571988, 6.51183]])  # N particles in a 10x10x10 box
velocities = np.array([[4.37599538, 8.91785328, 9.63675087],
 [3.83429192, 7.91712711, 5.28882593]])  # N particles in a 10x10x10 box
L = 10.0
sigma = 1.0
epsilon = 1.0
m = 1
dt = 0.01
rc = 5
assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
N = 5
positions = np.array([[4.23726683, 7.24497545, 0.05701276],
 [3.03736449, 1.48736913, 1.00346047],
 [1.9594282,  3.48694961, 4.03690693],
 [5.47580623, 4.28140578, 6.8606994 ],
 [2.04842798, 8.79815741, 0.36169018]])  # N particles in a 10x10x10 box
velocities = np.array([[6.70468142, 4.17305434, 5.5869046 ],
 [1.40388391, 1.98102942, 8.00746021],
 [9.68260045, 3.13422647, 6.92321085],
 [8.76388599, 8.9460611,  0.85043658],
 [0.39054783, 1.6983042,  8.78142503]])  # N particles in a 10x10x10 box
L = 10.0
sigma = 1.0
epsilon = 1.0
m = 1
dt = 0.01
rc = 5
assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
N = 10
positions = np.array([[4.4067278,  0.27943667, 5.56066548],
 [4.40153135, 4.25420214, 3.34203792],
 [2.12585237, 6.25071237, 3.01277888],
 [2.73834532, 6.30779077, 5.34141911],
 [1.43475136, 5.16994248, 1.90111297],
 [7.89610915, 8.58343073, 5.02002737],
 [8.51917219, 0.89182592, 5.10687864],
 [0.66107906, 4.31786204, 1.05039873],
 [1.31222278, 5.97016888, 2.28483329],
 [1.0761712,  2.30244719, 3.5953208 ]])  # N particles in a 10x10x10 box
velocities = np.array([[4.67788095, 2.01743837, 6.40407336],
 [4.83078905, 5.0524579,  3.86901721],
 [7.93680657, 5.80047382, 1.62341802],
 [7.00701382, 9.64500116, 4.99957397],
 [8.89518145, 3.41611733, 5.67142208],
 [4.27603192, 4.36804492, 7.76616414],
 [5.35546945, 9.53684998, 5.44150931],
 [0.8219327,  3.66440749, 8.50948852],
 [4.06178341, 0.27105663, 2.47080537],
 [0.67142725, 9.93850366, 9.70578668]])  # N particles in a 10x10x10 box
L = 10.0
sigma = 1.0
epsilon = 1.0
m = 1
dt = 0.01
rc = 5
assert cmp_tuple_or_list(velocity_verlet(N,sigma,epsilon,positions,velocities,dt,m,L,rc), target)
