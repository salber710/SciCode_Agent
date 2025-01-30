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
    if L <= 0:
        raise ValueError("Box size L must be greater than zero.")
    
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


# Background: In molecular dynamics simulations, the minimum image vector is used to determine the shortest vector between two particles in a periodic system. This vector is crucial for calculating forces and interactions between particles, as it accounts for the periodic boundary conditions. The minimum image vector is the displacement vector between two particles, adjusted to ensure it is the shortest possible vector within the periodic boundaries. For a cubic box of size L, this involves adjusting the displacement vector by subtracting L times the nearest integer to the displacement divided by L.


def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    '''
    if L <= 0:
        raise ValueError("Box size L must be greater than zero.")
    
    # Convert inputs to numpy arrays
    r1 = np.asarray(r1)
    r2 = np.asarray(r2)
    
    # Calculate the displacement vector
    delta = r2 - r1
    
    # Apply the minimum image convention to get the minimum image vector
    r12 = delta - L * np.round(delta / L)
    
    return r12


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. It is widely used in molecular dynamics simulations to model van der Waals forces. The potential is characterized by two parameters: epsilon (ε), which represents the depth of the potential well, and sigma (σ), which is the finite distance at which the inter-particle potential is zero. The Lennard-Jones potential is given by the formula:
# 
# V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6]
# 
# where r is the distance between the particles. The potential is attractive at long ranges and repulsive at short ranges. To improve computational efficiency, the potential is often truncated and shifted to zero at a cutoff distance rc. This means that for r > rc, the potential energy is set to zero, and for r <= rc, the potential is adjusted by subtracting the potential value at rc to ensure continuity.

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
    if r <= 0 or sigma <= 0 or epsilon <= 0 or rc <= 0:
        raise ValueError("All parameters r, sigma, epsilon, and rc must be positive numbers.")

    if r > rc:
        return 0.0
    else:
        # Calculate the Lennard-Jones potential
        sr6 = (sigma / r) ** 6
        lj_potential = 4 * epsilon * (sr6**2 - sr6)
        
        # Calculate the potential at the cutoff distance
        sr6_rc = (sigma / rc) ** 6
        lj_potential_rc = 4 * epsilon * (sr6_rc**2 - sr6_rc)
        
        # Truncate and shift the potential
        E = lj_potential - lj_potential_rc
        return E


# Background: The Lennard-Jones force is derived from the Lennard-Jones potential, which models the interaction between a pair of neutral atoms or molecules. The force is the negative gradient of the potential with respect to the distance between particles. For the Lennard-Jones potential, the force can be expressed as:
# F(r) = -dV/dr = 24 * ε * [(2 * (σ/r)^12) - ((σ/r)^6)] * (1/r)
# This force is attractive at long ranges and repulsive at short ranges. Similar to the potential, the force is often truncated and shifted to zero at a cutoff distance rc to improve computational efficiency. For r > rc, the force is set to zero. For r <= rc, the force is calculated using the above formula.


def f_ij(r_vector, sigma, epsilon, rc):
    '''Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r_vector (array_like): The displacement vector between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    '''
    r = np.linalg.norm(r_vector)
    if r == 0 or sigma <= 0 or epsilon <= 0 or rc <= 0:
        raise ValueError("All parameters r, sigma, epsilon, and rc must be positive numbers.")

    if r > rc:
        return np.zeros(3)  # Return zero force if beyond cutoff distance
    else:
        # Calculate the Lennard-Jones force
        sr6 = (sigma / r) ** 6
        force_magnitude = 24 * epsilon * (2 * sr6**2 - sr6) / r
        
        # The force vector is along the direction of the displacement vector
        r_hat = r_vector / r
        
        # Calculate the force vector
        f = force_magnitude * r_hat
        return f



# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff distance
# to improve computational efficiency. However, this truncation can lead to inaccuracies in the calculated potential
# energy of the system because it neglects the contributions from interactions beyond the cutoff distance. To account
# for these neglected interactions, a tail correction is applied. The tail correction for the energy is derived by
# integrating the Lennard-Jones potential from the cutoff distance to infinity, assuming a uniform particle density.
# The formula for the energy tail correction in a cubic box of volume V = L^3 with N particles is:
# E_tail = (8/3) * π * N * (N-1) * ρ * ε * [(σ/rc)^9 - 3*(σ/rc)^3]
# where ρ = N/V is the number density of the particles.





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
    if L <= 0 or sigma <= 0 or epsilon <= 0 or rc <= 0:
        raise ValueError("All parameters L, sigma, epsilon, and rc must be positive numbers.")
    
    # Calculate the volume of the cubic box
    V = L**3
    
    # Calculate the number density
    rho = N / V
    
    # Calculate the tail correction
    rc3 = (sigma / rc) ** 3
    rc9 = rc3 ** 3
    E_tail_LJ = (8.0 / 3.0) * math.pi * N * (N - 1) * rho * epsilon * (rc9 - 3 * rc3)
    
    # Convert the energy to zeptojoules (1 zJ = 10^-21 J)
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
