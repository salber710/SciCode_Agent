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
