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


# Background: In molecular simulations, when using the Lennard-Jones (LJ) potential, interactions are typically truncated
# beyond a cutoff distance (rc) to reduce computational cost. However, truncating the potential can lead to inaccuracies
# in the calculated properties, such as energy. Tail corrections are applied to account for the missing interactions
# beyond the cutoff distance. The tail correction for energy is derived from integrating the Lennard-Jones potential
# from the cutoff distance to infinity, considering a uniform distribution of particles beyond rc. This correction
# becomes significant in large systems and helps in providing more accurate thermodynamic properties. The energy tail
# correction per particle is given by:
# E_tail = (8/3) * π * ρ * ε * σ^3 * [(1/3)(σ/rc)^9 - (σ/rc)^3]
# where ρ is the number density, defined as N / V, with N being the number of particles and V the volume of the box (L^3).





def E_tail(N, L, sigma, epsilon, rc):
    '''Calculate the energy tail correction for a system of particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Lenght of cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    '''
    # Calculate the volume of the cubic box
    V = L**3
    # Calculate the number density
    rho = N / V
    # Calculate the cutoff ratio
    rc_ratio = sigma / rc
    # Calculate the energy tail correction per particle
    E_tail_per_particle = (8/3) * math.pi * rho * epsilon * sigma**3 * ((1/3) * rc_ratio**9 - rc_ratio**3)
    # Calculate the total energy tail correction for the system
    E_tail_total = N * E_tail_per_particle
    # Convert the result to zeptojoules
    E_tail_total_zeptojoules = E_tail_total * Avogadro * 1e-21
    return E_tail_total_zeptojoules


# Background: In molecular simulations, when truncating the Lennard-Jones (LJ) potential at a cutoff distance (rc),
# not only energy but also the calculated pressure can be affected. Tail corrections for pressure are applied to
# account for the missing contributions from interactions beyond rc. The pressure tail correction is derived from
# the virial theorem, and it considers the effect of the neglected forces on the pressure. The tail correction
# for pressure is calculated using the formula:
# P_tail = (16/3) * π * ρ^2 * ε * σ^3 * [(2/3)(σ/rc)^9 - (σ/rc)^3]
# where ρ is the number density of the system, defined as N / V, with N being the number of particles and V
# the volume of the simulation box (L^3). This correction becomes significant for large systems and helps provide
# more accurate pressure values.





def P_tail(N, L, sigma, epsilon, rc):
    ''' Calculate the pressure tail correction for a system of particles, including
    the truncated and shifted Lennard-Jones contributions.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Length of the cubic box.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The pressure tail correction for the entire system (in bar).
    '''
    # Calculate the volume of the cubic box
    V = L**3
    # Calculate the number density
    rho = N / V
    # Calculate the cutoff ratio
    rc_ratio = sigma / rc
    # Calculate the pressure tail correction per particle
    P_tail_per_particle = (16/3) * math.pi * rho**2 * epsilon * sigma**3 * ((2/3) * rc_ratio**9 - rc_ratio**3)
    # Calculate the total pressure tail correction for the system
    P_tail_total = N * P_tail_per_particle
    # Convert the result to bar (1 Pa = 1e-5 bar)
    P_tail_bar = P_tail_total * Avogadro * 1e-5
    return P_tail_bar


# Background: The total potential energy of a system of particles is crucial for understanding the interactions
# and stability of the system. In the context of molecular simulations using the Lennard-Jones (LJ) potential,
# the total potential energy is the sum of the pairwise potential energies between all particles, considering
# periodic boundary conditions and the truncated and shifted potential. For a system with N particles, the potential
# energy is calculated by summing the contributions from each unique pair of particles, which involves computing
# the distance between each pair, applying the minimum image convention to account for periodic boundaries, and
# using the truncated and shifted LJ potential to find the energy contribution from each pair. This ensures
# accurate representation of the interactions within the simulation box.




def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Length of cubic box.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float
        The total potential energy of the system (in zeptojoules).
    '''
    N = xyz.shape[0]
    E = 0.0

    # Iterate over each unique pair of particles
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the displacement vector between particles i and j
            delta = xyz[j] - xyz[i]
            # Apply the minimum image convention
            delta = delta - L * np.round(delta / L)
            # Calculate the distance between particles i and j
            r = np.linalg.norm(delta)

            # Compute the Lennard-Jones potential if within cutoff
            if r < rc:
                sr6 = (sigma / r) ** 6
                V_r = 4 * epsilon * (sr6**2 - sr6)
                # Calculate the potential at the cutoff distance rc
                sr6_cutoff = (sigma / rc) ** 6
                V_rc = 4 * epsilon * (sr6_cutoff**2 - sr6_cutoff)
                # Calculate the shifted potential
                E += V_r - V_rc

    # Convert the total energy to zeptojoules
    E_zeptojoules = E * Avogadro * 1e-21
    return E_zeptojoules



# Background: In molecular dynamics simulations, the instantaneous temperature of a system can be calculated using the kinetic energy of the particles. 
# According to the equipartition theorem, each degree of freedom contributes (1/2)kT to the average kinetic energy, where k is the Boltzmann constant, and T is the temperature. 
# For a system with N particles, each particle having three degrees of freedom (corresponding to motion in three-dimensional space), the total kinetic energy K of all particles is:
# K = (1/2) * m * sum(v_i^2)
# where v_i is the velocity of the i-th particle and m is the molar mass of the particles.
# The instantaneous temperature T can then be calculated using:
# T = (2 * K) / (3 * NkB)
# where kB is the Boltzmann constant in zJ/K (0.0138064852 zJ/K), and N is the number of particles in the system.

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
    # Convert the molar mass from g/mol to kg for calculations
    m_kg = m / 1000.0 / Avogadro  # kg per particle

    # Calculate the kinetic energy (in zeptojoules)
    # v_xyz is in nm/ps, so we convert this to m/s (1 nm/ps = 1e-12 m/s)
    velocities_m_s = v_xyz * 1e-12
    kinetic_energy = 0.5 * m_kg * np.sum(velocities_m_s**2)  # in zeptojoules

    # Boltzmann constant in zJ/K
    k_B = 0.0138064852

    # Calculate temperature using the equipartition theorem
    T = (2 * kinetic_energy) / (3 * N * k_B)

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
