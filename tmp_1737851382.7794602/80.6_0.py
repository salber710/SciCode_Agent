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
    elif r <= 0:
        # Handle non-positive distance as an error
        raise ValueError("Distance r must be positive and non-zero.")
    elif sigma <= 0:
        # Handle non-positive sigma as an error
        raise ValueError("Sigma must be positive and non-zero.")
    elif epsilon < 0:
        # Handle negative epsilon as an error
        raise ValueError("Epsilon must be non-negative.")
    else:
        # Calculate the magnitude of the Lennard-Jones force
        force_magnitude = 24 * epsilon * ((2 * (sigma / r)**12) - ((sigma / r)**6)) / r
        return force_magnitude


# Background: In molecular dynamics simulations, calculating the forces on each particle due to pairwise interactions
# is crucial for understanding the dynamics of the system. The force on a particle is derived from the negative gradient
# of the potential energy with respect to the particle's position. For the Lennard-Jones potential, the force between two
# particles is given by F_LJ(r) = 24 * ε * [(2 * (σ/r)^12) - ((σ/r)^6)] / r, where r is the distance between the particles.
# The force is truncated to zero at a cutoff distance rc to ensure computational efficiency. The net force on each particle
# is the vector sum of the forces due to all other particles, considering periodic boundary conditions using the minimum
# image convention.


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
    
    if N != xyz.shape[0]:
        raise ValueError("Mismatch between number of particles N and the number of position vectors in xyz.")
    
    def dist(r1, r2, L):
        '''Calculate the minimum image distance between two atoms in a periodic cubic system.'''
        dx = r2[0] - r1[0]
        dy = r2[1] - r1[1]
        dz = r2[2] - r1[2]
        
        dx -= L * round(dx / L)
        dy -= L * round(dy / L)
        dz -= L * round(dz / L)
        
        return np.sqrt(dx**2 + dy**2 + dz**2), np.array([dx, dy, dz])

    def f_ij(r, sigma, epsilon, rc):
        '''Calculate the magnitude of the Lennard-Jones force between two particles.'''
        if r >= rc or r == 0:
            return 0.0
        else:
            return 24 * epsilon * ((2 * (sigma / r)**12) - ((sigma / r)**6)) / r

    f_xyz = np.zeros((N, 3))

    for i in range(N):
        for j in range(i + 1, N):
            r, displacement = dist(xyz[i], xyz[j], L)
            force_magnitude = f_ij(r, sigma, epsilon, rc)
            if r != 0:  # Avoid division by zero
                force_vector = force_magnitude * (displacement / r)
                f_xyz[i] += force_vector
                f_xyz[j] -= force_vector  # Newton's third law: action = -reaction

    return f_xyz



# Background: The Velocity Verlet algorithm is a numerical method used to integrate Newton's equations of motion.
# It is particularly useful in molecular dynamics simulations for updating the positions and velocities of particles
# over time. The algorithm is symplectic, meaning it conserves the Hamiltonian structure of the equations, which
# helps maintain energy conservation over long simulations. The algorithm works by first updating the positions
# using the current velocities and accelerations, then calculating the new forces (and thus accelerations), and
# finally updating the velocities using the average of the old and new accelerations. This method is efficient and
# provides good stability for simulating systems with many interacting particles.

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
