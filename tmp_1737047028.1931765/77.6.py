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


# Background: The Lennard-Jones force is derived from the Lennard-Jones potential, which models the interaction
# between a pair of neutral atoms or molecules. The force is the negative gradient of the potential with respect
# to the distance between the particles. For the Lennard-Jones potential V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6],
# the force F(r) is given by F(r) = -dV/dr. This results in the expression:
# F(r) = 24 * ε * [(2 * (σ/r)^12) - ((σ/r)^6)] / r. The force is attractive when the particles are far apart
# and repulsive when they are close together. Similar to the potential, the force is truncated and shifted to
# zero at a cutoff distance rc. If the distance r is greater than or equal to rc, the force is considered to be zero.


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
    if r < rc:
        # Calculate the magnitude of the Lennard-Jones force
        force_magnitude = 24 * epsilon * ((2 * (sigma / r)**12) - ((sigma / r)**6)) / r
        # Return the force vector
        return force_magnitude
    else:
        # If the distance is greater than or equal to the cutoff, the force is zero
        return 0.0



# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff
# distance to improve computational efficiency. However, this truncation can lead to inaccuracies in the
# calculated potential energy of the system because it neglects the contributions from interactions beyond
# the cutoff distance. To account for these neglected interactions, a tail correction is applied to the
# potential energy. The tail correction for the energy is derived by integrating the Lennard-Jones potential
# from the cutoff distance to infinity, assuming a uniform distribution of particles. For a system of N
# particles in a cubic box of volume V = L^3, the tail correction to the energy is given by:
# E_tail = (8/3) * π * N * (N-1) * ρ * ε * [(σ/rc)^9 - 3*(σ/rc)^3],
# where ρ is the number density of the system, ρ = N/V. This correction is important for accurately
# estimating the total potential energy of the system.





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
    float: The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    '''
    # Calculate the volume of the cubic box
    V = L**3
    
    # Calculate the number density
    rho = N / V
    
    # Calculate the tail correction for the energy
    E_tail_LJ = (8/3) * math.pi * N * (N - 1) * rho * epsilon * ((sigma / rc)**9 - 3 * (sigma / rc)**3)
    
    # Convert the energy to zeptojoules (1 zeptojoule = 1e-21 joules)
    E_tail_LJ *= 1e21
    
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
