import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro

# Background: In computational simulations, especially those involving molecular dynamics or particle systems, it is common to use periodic boundary conditions (PBCs) to simulate an infinite system using a finite-sized simulation box. This approach helps to minimize edge effects and allows particles to move seamlessly across the boundaries of the simulation box. When a particle moves out of one side of the box, it re-enters from the opposite side. Mathematically, this is achieved by wrapping the particle's coordinates back into the box using the modulo operation. For a cubic box of size L, each coordinate of a particle is wrapped using the formula: wrapped_coordinate = coordinate % L. This ensures that all particle coordinates remain within the range [0, L).


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use numpy's modulo operation to wrap each coordinate within the range [0, L)
    coord = np.mod(r, L)
    return coord


# Background: In molecular simulations with periodic boundary conditions, the minimum image convention is used to calculate the shortest distance between two particles. This is crucial for accurately computing interactions in a periodic system. The minimum image distance is the smallest distance between two particles, considering that each particle can be represented by an infinite number of periodic images. For a cubic box of size L, the minimum image distance along each coordinate can be calculated by considering the displacement and adjusting it to be within the range [-L/2, L/2]. This ensures that the shortest path across the periodic boundaries is used.


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
    
    # Calculate the Euclidean distance using the adjusted displacement
    distance = np.linalg.norm(delta)
    
    return distance


# Background: In molecular simulations with periodic boundary conditions, it is often necessary to calculate not just the minimum image distance but also the minimum image vector between two particles. The minimum image vector is the vector that points from one particle to another, considering the periodic boundaries, and is adjusted to be the shortest possible vector. This is important for calculating forces and other vector quantities in a periodic system. For a cubic box of size L, the minimum image vector along each coordinate can be calculated by considering the displacement and adjusting it to be within the range [-L/2, L/2]. This ensures that the shortest path across the periodic boundaries is used.

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
    
    # Apply the minimum image convention to each component of the vector
    delta = delta - L * np.round(delta / L)
    
    return delta


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. It is widely used in molecular dynamics simulations to model van der Waals forces. The potential is characterized by two parameters: sigma (σ), which is the distance at which the potential is zero, and epsilon (ε), which is the depth of the potential well. The Lennard-Jones potential is given by the formula:
# 
# V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6]
# 
# where r is the distance between the particles. The potential is attractive at long ranges and repulsive at short ranges. In practice, the potential is often truncated and shifted to zero at a cutoff distance rc to improve computational efficiency. This means that for distances greater than rc, the potential energy is set to zero. The truncated and shifted Lennard-Jones potential is calculated as:
# 
# V_truncated(r) = V(r) - V(rc) for r < rc
# V_truncated(r) = 0 for r >= rc
# 
# This ensures that the potential smoothly goes to zero at the cutoff distance.

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
    if r >= rc:
        return 0.0
    else:
        # Calculate the Lennard-Jones potential at distance r
        sr6 = (sigma / r) ** 6
        lj_potential = 4 * epsilon * (sr6 ** 2 - sr6)
        
        # Calculate the Lennard-Jones potential at the cutoff distance rc
        src6 = (sigma / rc) ** 6
        lj_potential_rc = 4 * epsilon * (src6 ** 2 - src6)
        
        # Truncate and shift the potential
        E = lj_potential - lj_potential_rc
        return E


# Background: The Lennard-Jones force is derived from the Lennard-Jones potential, which models the interaction between a pair of neutral atoms or molecules. The force is the negative gradient of the potential with respect to the distance between the particles. For the Lennard-Jones potential, the force can be calculated using the formula:
# 
# F(r) = -dV/dr = 24 * ε * [(2 * (σ/r)^12) - ((σ/r)^6)] * (1/r)
# 
# where r is the distance between the particles, σ is the distance at which the potential is zero, and ε is the depth of the potential well. The force is attractive at long ranges and repulsive at short ranges. In practice, the force is often truncated and shifted to zero at a cutoff distance rc to improve computational efficiency. This means that for distances greater than rc, the force is set to zero. The force vector is directed along the line connecting the two particles, and its magnitude is given by the above formula.


def f_ij(r, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (array_like): The displacement vector between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    # Calculate the distance between the particles
    r_mag = np.linalg.norm(r)
    
    if r_mag >= rc:
        return np.zeros_like(r)
    else:
        # Calculate the force magnitude using the Lennard-Jones force formula
        sr2 = (sigma / r_mag) ** 2
        sr6 = sr2 ** 3
        force_magnitude = 24 * epsilon * (2 * sr6 ** 2 - sr6) / r_mag
        
        # Calculate the force vector
        force_vector = force_magnitude * (r / r_mag)
        
        return force_vector


# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff distance
# to improve computational efficiency. However, this truncation can lead to inaccuracies in the calculated potential
# energy of the system because it neglects the contributions from interactions beyond the cutoff distance. To account
# for these neglected interactions, a tail correction is applied. The tail correction for the energy is derived by
# integrating the Lennard-Jones potential from the cutoff distance to infinity, assuming a uniform particle density.
# For a system of N particles in a cubic box of volume V = L^3, the tail correction to the energy is given by:
# 
# E_tail = (8/3) * π * N * (N-1) * ρ * ε * [(σ/rc)^9 - 3*(σ/rc)^3]
# 
# where ρ = N/V is the number density of the particles. This correction is important for accurately estimating the
# total potential energy of the system, especially in dense systems where many interactions are truncated.

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
    # Calculate the number density
    rho = N / (L ** 3)
    
    # Calculate the tail correction using the formula
    rc3 = (sigma / rc) ** 3
    rc9 = rc3 ** 3
    E_tail_LJ = (8 / 3) * math.pi * N * (N - 1) * rho * epsilon * (rc9 - 3 * rc3)
    
    return E_tail_LJ



# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff distance
# to improve computational efficiency. However, this truncation can lead to inaccuracies in the calculated pressure
# of the system because it neglects the contributions from interactions beyond the cutoff distance. To account for
# these neglected interactions, a tail correction is applied to the pressure. The tail correction for the pressure
# is derived by integrating the virial contribution of the Lennard-Jones potential from the cutoff distance to infinity,
# assuming a uniform particle density. For a system of N particles in a cubic box of volume V = L^3, the tail correction
# to the pressure is given by:
# 
# P_tail = (16/3) * π * N * (N-1) * ρ^2 * ε * [(2/3)*(σ/rc)^9 - (σ/rc)^3]
# 
# where ρ = N/V is the number density of the particles. This correction is important for accurately estimating the
# total pressure of the system, especially in dense systems where many interactions are truncated.

def P_tail(N, L, sigma, epsilon, rc):
    ''' Calculate the pressure tail correction for a system of particles, including
     the truncated and shifted Lennard-Jones contributions.
    Parameters:
     N (int): The total number of particles in the system.
     L (float): Length of cubic box
     sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
     epsilon (float): The depth of the potential well for the Lennard-Jones potential.
     rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
     Returns:
     float
         The pressure tail correction for the entire system (in bar).
    '''
    # Calculate the number density
    rho = N / (L ** 3)
    
    # Calculate the tail correction using the formula
    rc3 = (sigma / rc) ** 3
    rc9 = rc3 ** 3
    P_tail_LJ = (16 / 3) * math.pi * N * (N - 1) * (rho ** 2) * epsilon * ((2/3) * rc9 - rc3)
    
    # Convert the pressure from the simulation units to bar
    # 1 atm = 101325 Pa, and 1 bar = 100000 Pa
    # Assuming the simulation units are in terms of energy per volume, we need to convert to bar
    P_tail_bar = P_tail_LJ * 1e-5  # Convert from Pa to bar
    
    return P_tail_bar


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('77.7', 3)
target = targets[0]

N=2
L=10
sigma = 1
epsilon = 1
rc = 1
assert np.allclose(P_tail(N,L,sigma,epsilon,rc), target)
target = targets[1]

N=5
L=10
sigma = 1
epsilon = 1
rc = 5
assert np.allclose(P_tail(N,L,sigma,epsilon,rc), target)
target = targets[2]

N=10
L=10
sigma = 1
epsilon = 1
rc = 9
assert np.allclose(P_tail(N,L,sigma,epsilon,rc), target)
