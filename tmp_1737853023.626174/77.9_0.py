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
    if L <= 0 or sigma <= 0 or rc <= 0:
        raise ValueError("Parameters L, sigma, and rc must be positive numbers.")
    
    if epsilon < 0:
        raise ValueError("Epsilon must be a non-negative number.")
    
    if N < 0:
        raise ValueError("Number of particles N must be non-negative.")
    
    if N < 2:
        return 0.0  # No interactions possible with less than two particles

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


# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff distance
# to improve computational efficiency. This truncation can lead to inaccuracies in the calculated pressure of the system
# because it neglects the contributions from interactions beyond the cutoff distance. To account for these neglected
# interactions, a tail correction is applied. The tail correction for the pressure is derived by integrating the
# Lennard-Jones force from the cutoff distance to infinity, assuming a uniform particle density. The formula for the
# pressure tail correction in a cubic box of volume V = L^3 with N particles is:
# P_tail = (16/3) * π * N * (N-1) * ρ^2 * ε * [(2/3)*(σ/rc)^9 - (σ/rc)^3]
# where ρ = N/V is the number density of the particles. The result is typically converted to bar for practical use.





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
    if L <= 0 or sigma <= 0 or rc <= 0:
        raise ValueError("Parameters L, sigma, and rc must be positive numbers.")
    
    if epsilon < 0:
        raise ValueError("Epsilon must be a non-negative number.")
    
    if N < 0:
        raise ValueError("Number of particles N must be non-negative.")
    
    if N < 2:
        return 0.0  # No interactions possible with less than two particles

    # Calculate the volume of the cubic box
    V = L**3
    
    # Calculate the number density
    rho = N / V
    
    # Calculate the tail correction for pressure
    rc3 = (sigma / rc) ** 3
    rc9 = rc3 ** 3
    P_tail_LJ = (16.0 / 3.0) * math.pi * N * (N - 1) * rho**2 * epsilon * ((2.0 / 3.0) * rc9 - rc3)
    
    # Convert the pressure to bar (1 Pa = 1e-5 bar)
    P_tail_bar = P_tail_LJ * 1e-5
    
    return P_tail_bar


# Background: In molecular dynamics simulations, the total potential energy of a system of particles is a crucial
# quantity that helps in understanding the interactions and stability of the system. The Lennard-Jones potential
# is commonly used to model these interactions, especially for neutral atoms and molecules. The total potential
# energy is calculated by summing the pairwise Lennard-Jones potential energies between all unique pairs of particles
# in the system. The potential is truncated and shifted to zero at a cutoff distance `rc` to improve computational
# efficiency. The potential energy is calculated using the formula:
# V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6] for r <= rc, and V(r) = 0 for r > rc.
# The total potential energy is the sum of these pairwise interactions, and it is often expressed in zeptojoules
# (1 zJ = 10^-21 J) for practical purposes.





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
    if xyz.size == 0:
        return 0.0

    N = xyz.shape[0]
    total_energy = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the minimum image vector between particles i and j
            r_vector = xyz[j] - xyz[i]
            r_vector = r_vector - L * np.round(r_vector / L)
            
            # Calculate the distance
            r = np.linalg.norm(r_vector)
            
            # Calculate the Lennard-Jones potential energy for this pair
            if r <= rc:
                sr6 = (sigma / r) ** 6
                lj_potential = 4 * epsilon * (sr6**2 - sr6)
                
                # Calculate the potential at the cutoff distance
                sr6_rc = (sigma / rc) ** 6
                lj_potential_rc = 4 * epsilon * (sr6_rc**2 - sr6_rc)
                
                # Truncate and shift the potential
                E_ij = lj_potential - lj_potential_rc
                total_energy += E_ij

    # Convert the total energy to zeptojoules (1 zJ = 10^-21 J)
    total_energy *= 1e21

    return total_energy



# Background: In molecular dynamics simulations, the instantaneous temperature of a system can be calculated using the equipartition theorem. 
# According to this theorem, the average kinetic energy of a system is related to its temperature. For a system of N particles, each with 
# three degrees of freedom (corresponding to motion in the x, y, and z directions), the total kinetic energy is given by:
# KE_total = (1/2) * m * sum(v_i^2) for i = 1 to N, where v_i is the velocity of particle i.
# The equipartition theorem states that the average kinetic energy per degree of freedom is (1/2) * k_B * T, where k_B is the Boltzmann constant.
# Therefore, the instantaneous temperature T can be calculated as:
# T = (2/3) * (KE_total / (N * k_B))
# The Boltzmann constant k_B is given as 0.0138064852 zJ/K. The velocities are provided in nm/ps, and the molar mass m is in g/mol.
# To convert the mass to kg, we use the relation: 1 g/mol = 1e-3 kg/mol, and divide by Avogadro's number to get the mass of a single particle.

def temperature(v_xyz, m, N):
    '''Calculate the instantaneous temperature of a system of particles using the equipartition theorem.
    Parameters:
    v_xyz : ndarray
        A NumPy array with shape (N, 3) containing the velocities of each particle in the system,
        in nanometers per picosecond (nm/ps).
    m : float
        The molar mass of the particles in the system, in grams per mole (g/mol).
    N : int
        The number of particles in the system.
    Returns:
    float
        The instantaneous temperature of the system in Kelvin (K).
    '''
    # Convert molar mass from g/mol to kg
    m_kg = m * 1e-3 / Avogadro
    
    # Calculate the kinetic energy
    # KE_total = (1/2) * m * sum(v_i^2) for all particles
    v_squared = np.sum(v_xyz**2, axis=1)  # Sum of squares of velocities for each particle
    KE_total = 0.5 * m_kg * np.sum(v_squared)  # Total kinetic energy
    
    # Boltzmann constant in zeptojoules per Kelvin
    k_B = 0.0138064852  # zJ/K
    
    # Calculate the temperature using the equipartition theorem
    T = (2.0 / 3.0) * (KE_total / (N * k_B))
    
    return T

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('77.9', 3)
target = targets[0]

v=np.array([1,2,3])
m=1
N=1
assert np.allclose(temperature(v,m,N), target)
target = targets[1]

v=np.array([[1,2,3],[1,1,1]])
m=10
N=2
assert np.allclose(temperature(v,m,N), target)
target = targets[2]

v=np.array([[1,2,3],[4,6,8],[6,1,4]])
m=100
N=3
assert np.allclose(temperature(v,m,N), target)
