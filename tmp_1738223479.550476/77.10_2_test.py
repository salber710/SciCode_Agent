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




def temperature(v_xyz: List[List[float]], m: float, N: int) -> float:
    '''Calculate the instantaneous temperature of a system of particles using the equipartition theorem.
    Parameters:
    v_xyz : List of Lists
        A list of lists with length N, each containing 3 floats, representing the velocities of each particle 
        in the system, in nanometers per picosecond (nm/ps).
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
    avogadro_number = 6.02214076e23  # Avogadro's number
    mass_per_particle_kg = m / (avogadro_number * 1000)  # g/mol to kg/particle

    # Convert velocities from nm/ps to m/s
    factor_nm_ps_to_m_s = 1e3  # nm/ps to m/s
    velocities_m_s = [[vel * factor_nm_ps_to_m_s for vel in particle] for particle in v_xyz]
    
    # Calculate total kinetic energy using a generator expression
    kinetic_energy = sum(0.5 * mass_per_particle_kg * (vx**2 + vy**2 + vz**2) for vx, vy, vz in velocities_m_s)
    
    # Calculate temperature using the equipartition theorem
    temperature_kelvin = (2 * kinetic_energy) / (3 * N * k_B)
    
    return temperature_kelvin




def pressure(N, L, T, xyz, sigma, epsilon, rc):
    # Boltzmann constant in zeptojoules per Kelvin
    k_B = 0.0138064852  # zJ/K

    # Calculate the volume in cubic meters
    V_m3 = (L * 1e-9) ** 3

    # Calculate the number density (particles per cubic meter)
    num_density = N / V_m3

    # Calculate the kinetic pressure using the ideal gas law
    P_kinetic = num_density * k_B * T * 1e3  # Convert from zJ/m^3 to bar

    # Initialize the virial sum
    virial_accumulation = 0.0

    # Iterate over each pair of particles
    for i in range(N):
        for j in range(i + 1, N):
            # Compute the distance vector between particles i and j
            delta_r = xyz[j] - xyz[i]

            # Apply periodic boundary conditions
            delta_r -= L * np.round(delta_r / L)

            # Calculate the magnitude of the distance vector
            distance = np.sqrt(np.sum(delta_r ** 2))

            if distance < rc:
                # Calculate the inverse distance powers for Lennard-Jones potential
                inv_dist = sigma / distance
                inv_dist_6 = inv_dist ** 6
                inv_dist_12 = inv_dist_6 ** 2

                # Calculate the force magnitude as per Lennard-Jones potential
                lj_force = 48 * epsilon * (inv_dist_12 - 0.5 * inv_dist_6) / distance

                # Accumulate the virial term
                virial_accumulation += np.dot(delta_r, delta_r) * lj_force

    # Calculate the virial pressure contribution
    P_virial = virial_accumulation / (3 * V_m3) * 1e-5  # Convert from zJ/m^3 to bar

    # Return the kinetic, virial, and total pressure
    return P_kinetic, P_virial, P_kinetic + P_virial


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e