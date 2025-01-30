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
    '''Compute the tail correction energy for a system of particles using a distinct approach
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
    
    # Volume of the cubic box
    V = L**3
    
    # Calculate reduced cutoff distance
    x = rc / sigma
    
    # Use a different method to calculate powers of x
    x3 = pow(x, 3)
    x9 = pow(x, 9)
    
    # Calculate the integral term using a different expression
    integral_term = (x9 - 3 * x3) / 9
    
    # Calculate the energy tail correction
    # Use an approximation for pi to avoid importing any library
    pi_approx = 22 / 7
    energy_correction = (8 * pi_approx * epsilon * sigma**3 * N * (N + 1) * integral_term) / (3 * V)
    
    return energy_correction


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e