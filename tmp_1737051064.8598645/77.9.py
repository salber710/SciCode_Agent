import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro
def wrap(r, L):
    """Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    """
    coord = np.mod(r, L)
    return coord
def dist(r1, r2, L):
    """Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    """
    delta = np.array(r2) - np.array(r1)
    delta = delta - L * np.round(delta / L)
    distance = np.linalg.norm(delta)
    return distance
def dist_v(r1, r2, L):
    """Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    numpy.ndarray: The minimum image vector between the two atoms.
    """
    delta = np.array(r2) - np.array(r1)
    delta = delta - L * np.round(delta / L)
    return delta
def E_ij(r, sigma, epsilon, rc):
    """Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The combined potential energy between the two particles, considering the specified potentials.
    """
    if r < rc:
        V_r = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        V_rc = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        return V_r - V_rc
    else:
        return 0.0
def f_ij(r, sigma, epsilon, rc):
    """Calculate the force vector between two particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    r (float): The distance between particles i and j.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials
    """
    if r < rc:
        force_magnitude = 24 * epsilon * (2 * (sigma / r) ** 12 - (sigma / r) ** 6) / r
        return force_magnitude
    else:
        return 0.0
def E_tail(N, L, sigma, epsilon, rc):
    """Calculate the energy tail correction for a system of particles, considering the truncated and shifted
    Lennard-Jones potential.
    Parameters:
    N (int): The total number of particles in the system.
    L (float): Length of cubic box.
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.
    """
    V = L ** 3
    rho = N / V
    E_tail_LJ = 8 / 3 * math.pi * N * (N - 1) * rho * epsilon * ((sigma / rc) ** 9 - 3 * (sigma / rc) ** 3)
    E_tail_LJ *= 1e+21
    return E_tail_LJ

# Background: In molecular dynamics simulations, the Lennard-Jones potential is often truncated at a cutoff distance
# to reduce computational cost. However, this truncation can lead to inaccuracies in calculated properties such as
# pressure. To account for the effects of truncation, tail corrections are applied. The pressure tail correction
# accounts for the contribution of the interactions beyond the cutoff distance. This correction is derived from
# integrating the Lennard-Jones potential over the volume outside the cutoff sphere. The correction depends on the
# number density of the particles, the cutoff distance, and the parameters of the Lennard-Jones potential (sigma and
# epsilon). The pressure tail correction is typically expressed in units of pressure, such as bar.





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
    V = L ** 3
    rho = N / V
    P_tail_LJ = (16 / 3) * math.pi * rho**2 * epsilon * ((2/3) * (sigma / rc)**9 - (sigma / rc)**3)
    P_tail_LJ *= 1e-5 * Avogadro  # Convert from Pa to bar using Avogadro's number
    return P_tail_LJ


# Background: In molecular dynamics simulations, the potential energy of a system of particles is a crucial quantity
# that determines the system's stability and behavior. The Lennard-Jones potential is commonly used to model the
# interactions between particles. It accounts for the attractive and repulsive forces between particles based on
# their distance. The potential energy of a system is calculated by summing the pairwise Lennard-Jones potentials
# between all particles, considering periodic boundary conditions and a cutoff distance to truncate and shift the
# potential to zero beyond this distance. This ensures computational efficiency while maintaining accuracy.





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
    N = xyz.shape[0]
    E = 0.0

    for i in range(N):
        for j in range(i + 1, N):
            r1 = xyz[i]
            r2 = xyz[j]
            # Calculate the minimum image distance
            delta = r2 - r1
            delta = delta - L * np.round(delta / L)
            r = np.linalg.norm(delta)
            
            # Calculate the Lennard-Jones potential energy for this pair
            if r < rc:
                V_r = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
                V_rc = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
                E += V_r - V_rc

    # Convert energy to zeptojoules
    E *= 1e+21
    return E



# Background: In molecular dynamics simulations, the temperature of a system can be calculated using the equipartition theorem, which relates the kinetic energy of the particles to the temperature. The kinetic energy of a system of particles is given by the sum of the kinetic energies of all individual particles. For a system of N particles, each with mass m and velocity v, the total kinetic energy (KE) is (1/2) * m * sum(v_i^2) for all particles i. According to the equipartition theorem, the average kinetic energy per degree of freedom is (1/2) * k_B * T, where k_B is the Boltzmann constant and T is the temperature. In three dimensions, each particle has three degrees of freedom, so the total kinetic energy is (3/2) * N * k_B * T. Solving for T gives T = (2/3) * KE / (N * k_B). The Boltzmann constant k_B is 0.0138064852 zJ/K.

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
    m_kg = m / 1000 / Avogadro  # kg per particle

    # Calculate the kinetic energy
    v_squared = np.sum(v_xyz**2, axis=1)  # Sum of squares of velocities for each particle
    KE = 0.5 * m_kg * np.sum(v_squared)  # Total kinetic energy in kg*(nm/ps)^2

    # Convert kinetic energy to zeptojoules (1 kg*(nm/ps)^2 = 1e-21 J)
    KE_zJ = KE * 1e-21

    # Boltzmann constant in zeptojoules per Kelvin
    k_B = 0.0138064852

    # Calculate temperature using the equipartition theorem
    T = (2 / 3) * KE_zJ / (N * k_B)

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
