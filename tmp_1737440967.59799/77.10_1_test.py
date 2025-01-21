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


# Background: In molecular dynamics simulations, the instantaneous temperature of a system can be calculated
# using the equipartition theorem, which relates the kinetic energy of the particles to the temperature.
# According to the equipartition theorem, each degree of freedom contributes (1/2)k_B*T to the kinetic energy,
# where k_B is the Boltzmann constant. For a system of N particles in three dimensions, there are 3N degrees
# of freedom (ignoring constraints), and the total kinetic energy is given by (1/2) * m * v^2 for each particle,
# where m is the mass of a single particle and v is its velocity. The relationship for temperature T in Kelvin
# can thus be derived from the total kinetic energy: KE_total = (3/2) * N * k_B * T. Rearranging gives:
# T = (2/3) * (KE_total / (N * k_B)), where KE_total is the sum of the kinetic energies of all particles.
# The Boltzmann constant k_B = 0.0138064852 zJ/K.

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
    m_kg = m / 1000 / Avogadro

    # Calculate the kinetic energy in zJ
    kinetic_energy = 0.5 * m_kg * np.sum(v_xyz**2)

    # Boltzmann constant in zJ/K
    k_B = 0.0138064852

    # Calculate temperature in Kelvin
    T = (2 / 3) * kinetic_energy / (N * k_B)

    return T



# Background: In molecular simulations, the pressure of a system can be calculated using the virial equation,
# which relates the pressure to both the kinetic energy of the particles and their interactions as described
# by the potential energy. For a system of N particles in a cubic box of length L, the pressure can be divided 
# into two components: the kinetic pressure and the virial pressure. The kinetic pressure is related to the 
# temperature of the system and is given by the ideal gas law: P_kinetic = (N * k_B * T) / V, where V is the 
# volume of the box and k_B is the Boltzmann constant. The virial pressure accounts for the interactions between 
# particles and is calculated from the forces acting between particles. In the context of the Lennard-Jones 
# potential, the virial contribution is derived from the pairwise interactions and involves summing the dot product 
# of the displacement vectors and the forces over all pairs of particles. The total pressure is the sum of the 
# kinetic and virial pressures. It is important to apply periodic boundary conditions and the minimum image 
# convention to properly account for interactions across the box boundaries.

def pressure(N, L, T, xyz, sigma, epsilon, rc):
    '''Calculate the pressure of a system of particles using the virial theorem, considering
    the Lennard-Jones contributions.
    Parameters:
    N : int
        The number of particles in the system.
    L : float
        The length of the side of the cubic simulation box (in nanometers).
    T : float
        The instantaneous temperature of the system (in Kelvin).
    xyz : ndarray
        A NumPy array with shape (N, 3) containing the positions of each particle in the system, in nanometers.
    sigma : float
        The Lennard-Jones size parameter (in nanometers).
    epsilon : float
        The depth of the potential well (in zeptojoules).
    rc : float
        The cutoff distance beyond which the inter-particle potential is considered to be zero (in nanometers).
    Returns:
    tuple
        The kinetic pressure (in bar), the virial pressure (in bar), and the total pressure (kinetic plus virial, in bar) of the system.
    '''
    # Calculate the volume of the cubic box
    V = L**3
    
    # Boltzmann constant in zJ/K
    k_B = 0.0138064852
    
    # Calculate the kinetic pressure in bar
    P_kinetic = (N * k_B * T) / V * 1e-5  # Conversion from Pa to bar

    # Initialize the virial contribution
    virial_sum = 0.0

    # Iterate over each unique pair of particles
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the displacement vector between particles i and j
            delta = xyz[j] - xyz[i]
            # Apply the minimum image convention
            delta = delta - L * np.round(delta / L)
            # Calculate the distance between particles i and j
            r = np.linalg.norm(delta)

            # If within the cutoff, compute the force
            if r < rc:
                # Calculate (sigma / r)^6
                sr6 = (sigma / r) ** 6
                # Calculate the magnitude of the Lennard-Jones force
                force_magnitude = 24 * epsilon * (2 * sr6**2 - sr6) / r
                
                # Add the contribution to the virial sum
                virial_sum += np.dot(delta, force_magnitude * delta / r)

    # Calculate the virial pressure in bar
    P_virial = virial_sum / (3 * V) * 1e-5  # Conversion from Pa to bar

    # Return the kinetic pressure, virial pressure, and total pressure
    return P_kinetic, P_virial, P_kinetic + P_virial

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('77.10', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
N = 2
L = 10
sigma = 1
epsilon = 1
positions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])
rc = 1
T=300
assert cmp_tuple_or_list(pressure(N, L, T, positions, sigma, epsilon, rc), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
N = 2
L = 10
sigma = 1
epsilon = 1
positions = np.array([[.62726631, 5.3077771 , 7.29719649],
       [2.25031287, 8.58926428, 4.71262908],
          [3.62726631, 1.3077771 , 2.29719649]])
rc = 2
T=1
assert cmp_tuple_or_list(pressure(N, L, T, positions, sigma, epsilon, rc), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
N = 5
L = 10
sigma = 1
epsilon = 1
positions = np.array([[.62726631, 5.3077771 , 7.29719649],
       [7.25031287, 7.58926428, 2.71262908],
       [8.7866416 , 3.73724676, 9.22676027],
       [0.89096788, 5.3872004 , 7.95350911],
       [6.068183  , 3.55807037, 2.7965242 ]])
rc = 3
T=200
assert cmp_tuple_or_list(pressure(N, L, T, positions, sigma, epsilon, rc), target)
