from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import math
import numpy as np
import scipy as sp
from scipy.constants import  Avogadro

def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use a while loop to iteratively adjust each coordinate to lie within the range [0, L).
    wrapped_coords = []
    for ri in r:
        while ri >= L:
            ri -= L
        while ri < 0:
            ri += L
        wrapped_coords.append(ri)
    return wrapped_coords


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    
    # Calculate the difference vector between r2 and r1
    delta = [(r2[i] - r1[i]) for i in range(3)]
    
    # Apply the minimum image convention by checking the shortest distance directly
    for i in range(3):
        if delta[i] > L / 2:
            delta[i] -= L
        elif delta[i] <= -L / 2:
            delta[i] += L
    
    # Compute and return the Euclidean distance using a generator expression
    return sum((d ** 2 for d in delta)) ** 0.5


def dist_v(r1, r2, L):
    '''Calculate the minimum image vector between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    r12: tuple of floats, the minimum image vector between the two atoms.
    '''
    # Calculate the difference vector and apply minimum image convention using a mathematical function
    def min_image_component(d, L):
        # Use mathematical expression to adjust components
        return d - L * round(d / L)

    # Calculate each component using the helper function
    delta_x = min_image_component(r2[0] - r1[0], L)
    delta_y = min_image_component(r2[1] - r1[1], L)
    delta_z = min_image_component(r2[2] - r1[2], L)

    # Return the minimum image vector as a tuple
    return (delta_x, delta_y, delta_z)


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
    
    # Using Taylor series expansion for exponential approximation
    def lj_potential_taylor(distance, sig, eps):
        x = sig / distance
        x2 = x * x
        # Approximate (1 - x^6) and (1 - x^12) using Taylor series
        term_6 = 1 - x2 * (1 + x2 * (1 + x2))
        term_12 = 1 - x2 * x2 * (1 + x2 * (1 + x2))
        return 4 * eps * (term_12 - term_6)

    potential_r = lj_potential_taylor(r, sigma, epsilon)
    potential_rc = lj_potential_taylor(rc, sigma, epsilon)

    return potential_r - potential_rc



def f_ij(r, sigma, epsilon, rc):
    '''Calculate the Lennard-Jones force vector using a novel approach.
    Parameters:
    r (array_like): Three-dimensional displacement vector between particles i and j.
    sigma (float): Distance at which the inter-particle potential is zero.
    epsilon (float): Depth of the potential well.
    rc (float): Cutoff distance beyond which the potential is zero.
    Returns:
    array_like: Force vector on particle i due to particle j.
    '''
    r_magnitude = np.linalg.norm(r)
    
    if r_magnitude >= rc:
        return np.zeros_like(r)
    
    # Calculate the inverse distance and its powers
    inv_r = 1.0 / r_magnitude
    s = sigma * inv_r
    s4 = s * s * s * s
    s10 = s4 * s4 * s * s

    # Calculate the force magnitude using a new distinct formulation
    force_magnitude = 48 * epsilon * s10 * (s4 - 0.5) * inv_r * inv_r
    
    # Compute the force vector
    force_vector = force_magnitude * r
    
    return force_vector


def E_tail(N, L, sigma, epsilon, rc):
    '''Calculate the energy tail correction for a system of particles using a unique approach
    for the Lennard-Jones potential within a cubic simulation box.
    
    Parameters:
    N (int): Total number of particles in the system.
    L (float): Length of the cubic simulation box.
    sigma (float): Lennard-Jones parameter for zero potential distance.
    epsilon (float): Lennard-Jones parameter for potential well depth.
    rc (float): Cutoff distance for potential truncation.
    
    Returns:
    float: Tail correction energy for the system (in zeptojoules).
    '''
    
    # Calculate the volume of the simulation box
    volume = L**3
    
    # Dimensionless cutoff distance
    reduced_rc = rc / sigma
    
    # Calculate powers using a different sequence
    reduced_rc_power_2 = reduced_rc ** 2
    reduced_rc_power_6 = reduced_rc_power_2 ** 3
    
    # Create a custom integral term using a different mathematical formulation
    integral_term = (4/15) * reduced_rc_power_6 - (4/5) * reduced_rc_power_2
    
    # Use an approximation for pi to avoid importing libraries
    pi_approx = 3.14159
    
    # Calculate the energy tail correction
    energy_tail_correction = (24 * pi_approx * epsilon * sigma**3 * N * (N * 0.5) * integral_term) / (3 * volume)
    
    return energy_tail_correction


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

    # Calculate the volume of the cubic box using a different approach
    volume = L * L * L
    
    # Calculate the reduced cutoff distance
    reduced_rc = rc / sigma

    # Calculate a unique set of powers
    reduced_rc_19 = reduced_rc ** 19
    reduced_rc_21 = reduced_rc_19 * reduced_rc ** 2
    
    # Use an alternative value for pi
    pi_unique = 3.1415926535898
    
    # Calculate the pressure tail correction with a unique formula
    pressure_tail_correction = (
        (88 / 27) * pi_unique * epsilon * sigma ** 3 * N ** 2 *
        (reduced_rc_21 - 9 * reduced_rc_19) / volume
    )
    
    # Convert the pressure tail correction from internal units to bar using a unique conversion factor
    P_tail_bar = pressure_tail_correction * 1.6e-5
    
    return P_tail_bar




@jit(nopython=True)
def E_pot(xyz, L, sigma, epsilon, rc):
    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.
    Parameters:
    xyz : A NumPy array with shape (N, 3) where N is the number of particles. Each row contains the x, y, z coordinates of a particle in the system.
    L (float): Length of the cubic box
    sigma (float): The distance at which the inter-particle potential is zero for the Lennard-Jones potential.
    epsilon (float): The depth of the potential well for the Lennard-Jones potential.
    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.
    Returns:
    float: The total potential energy of the system (in zeptojoules).
    '''

    # Use a different method to apply periodic boundary conditions
    def periodic_wrap(d, L):
        return d - L * np.floor(d / L + 0.5)

    # Calculate the potential energy using a different numerical approach
    def lj_pot(r, sigma, epsilon, rc):
        if r >= rc:
            return 0.0
        inv_r2 = (sigma / r) ** 2
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        rc2 = (sigma / rc) ** 2
        rc6 = rc2 ** 3
        rc12 = rc6 ** 2
        return 4 * epsilon * (inv_r12 - inv_r6 - rc12 + rc6)

    N = xyz.shape[0]
    total_energy = 0.0

    # Iterate over all pairs by using a double loop and vectorized operations
    for i in range(N):
        for j in range(i + 1, N):
            dx = periodic_wrap(xyz[j, 0] - xyz[i, 0], L)
            dy = periodic_wrap(xyz[j, 1] - xyz[i, 1], L)
            dz = periodic_wrap(xyz[j, 2] - xyz[i, 2], L)
            r_ij = np.sqrt(dx**2 + dy**2 + dz**2)
            total_energy += lj_pot(r_ij, sigma, epsilon, rc)

    return total_energy




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
    
    # Boltzmann constant in zeptojoules per Kelvin
    k_B = 0.0138064852  # zJ/K

    # Convert molar mass from g/mol to kg/particle
    # 1 gram/mole = 1e-3 kg/mole and we divide by Avogadro's number to get kg/particle
    avogadro_number = 6.02214076e23  # Avogadro's number
    m_kg_per_particle = m / (avogadro_number * 1e3)  # kg/particle

    # Convert velocities from nm/ps to m/s
    v_m_s = v_xyz * 1e3  # 1 nm/ps = 1e-9 m/1e-12 s = 1e3 m/s

    # Compute the squared velocity magnitudes
    v_squared = np.sum(v_m_s**2, axis=1)

    # Calculate total kinetic energy
    total_kinetic_energy = np.sum(0.5 * m_kg_per_particle * v_squared)

    # Calculate temperature using the equipartition theorem
    T = (2 * total_kinetic_energy) / (3 * N * k_B)

    return T


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e