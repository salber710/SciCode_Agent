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

    # Create a function to apply periodic boundary conditions using numpy vector operations
    def apply_pbc_vectorized(xyz, L):
        return xyz - L * np.round(xyz / L)

    # Function to calculate the truncated and shifted Lennard-Jones potential using numpy operations
    def lj_potential_vectorized(r, sigma, epsilon, rc):
        r6 = (sigma / r)**6
        r12 = r6**2
        rc6 = (sigma / rc)**6
        rc12 = rc6**2
        potential_r = 4 * epsilon * (r12 - r6)
        potential_rc = 4 * epsilon * (rc12 - rc6)
        return np.where(r < rc, potential_r - potential_rc, 0.0)

    # Number of particles
    N = xyz.shape[0]
    
    # Initialize total energy to zero
    total_energy = 0.0
    
    # Vectorized computation of pairwise distances and energies
    for i in range(N - 1):
        # Calculate displacement vectors for all particles with index > i
        displacements = xyz[i+1:] - xyz[i]
        
        # Apply periodic boundary conditions
        displacements = apply_pbc_vectorized(displacements, L)
        
        # Calculate pairwise distances
        distances = np.linalg.norm(displacements, axis=1)
        
        # Accumulate potential energies
        total_energy += np.sum(lj_potential_vectorized(distances, sigma, epsilon, rc))
    
    return total_energy


try:
    targets = process_hdf5_to_tuple('77.8', 3)
    target = targets[0]
    positions1 = np.array([[1, 1, 1], [1.1, 1.1, 1.1]])
    L1 = 10.0
    sigma1 = 1.0
    epsilon1 = 1.0
    rc=5
    assert np.allclose(E_pot(positions1, L1, sigma1, epsilon1,rc), target)

    target = targets[1]
    positions2 = np.array([[1, 1, 1], [1, 9, 1], [9, 1, 1], [9, 9, 1]])
    L2 = 10.0
    sigma2 = 1.0
    epsilon2 = 1.0
    rc=5
    assert np.allclose(E_pot(positions2, L2, sigma2, epsilon2,rc), target)

    target = targets[2]
    np.random.seed(0)
    positions3 = np.random.rand(10, 3) * 10  # 10 particles in a 10x10x10 box
    L3 = 10.0
    sigma3 = 1.0
    epsilon3 = 1.0
    rc=5
    assert np.allclose(E_pot(positions3, L3, sigma3, epsilon3,rc), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e