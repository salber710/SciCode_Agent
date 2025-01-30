import numpy as np

# Background: In computational simulations, especially those involving particles in a confined space, it is common to use periodic boundary conditions (PBCs). 
# PBCs are used to simulate an infinite system by wrapping particles that move out of the simulation box back into the box. 
# This is done by taking the modulo of the particle's position with respect to the box size. 
# For a cubic box of size L, if a particle's coordinate exceeds L or is less than 0, it is wrapped around to stay within the range [0, L). 
# This ensures that the simulation space behaves as if it is infinite and continuous.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    if L <= 0:
        raise ValueError("Box size L must be positive and non-zero.")
    if np.isnan(L):
        raise ValueError("Box size L cannot be NaN.")
    if not isinstance(r, np.ndarray):
        raise TypeError("Input coordinates must be a numpy array.")
    if not np.issubdtype(r.dtype, np.number):
        raise TypeError("Input coordinates must be numeric.")
    # Use numpy's modulo operation to wrap the coordinates
    coord = np.mod(r, L)
    # Handle negative values correctly
    coord = np.where(coord < 0, coord + L, coord)
    return coord



# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model the forces between particles. The potential is defined as:
# V(r) = 4 * epsilon * [(sigma / r)^12 - (sigma / r)^6]
# where r is the distance between two particles, epsilon is the depth of the potential well, and sigma is the finite distance at which the inter-particle potential is zero.
# In a periodic system, the minimum image convention is used to calculate the shortest distance between particles, considering the periodic boundaries.
# The potential is only calculated if the distance between particles is less than a specified cut-off distance, r_c, to improve computational efficiency.


def E_i(r, pos, sigma, epsilon, L, r_c):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array, the (x, y, z) coordinates of the target particle.
    pos : An array of (x, y, z) coordinates for each of the other particles in the system.
    sigma : float, the distance at which the potential minimum occurs
    epsilon : float, the depth of the potential well
    L : float, the length of the side of the cubic box
    r_c : float, cut-off distance
    Returns:
    float, the total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''

    def E_ij(r_i, r_j, sigma, epsilon, L):
        '''Calculate the Lennard-Jones potential energy between two particles using minimum image convention.'''
        # Calculate the distance vector considering periodic boundaries
        delta = r_j - r_i
        delta = delta - L * np.round(delta / L)  # Minimum image convention
        # Calculate the distance
        r_ij = np.linalg.norm(delta)
        # Calculate the Lennard-Jones potential if within the cut-off distance
        if r_ij < r_c:
            sr6 = (sigma / r_ij) ** 6
            sr12 = sr6 ** 2
            return 4 * epsilon * (sr12 - sr6)
        else:
            return 0.0

    # Initialize total energy
    E = 0.0
    # Calculate the total energy by summing the pairwise interactions
    for r_j in pos:
        E += E_ij(r, r_j, sigma, epsilon, L)

    return E

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('60.2', 3)
target = targets[0]

r1 = np.array([0.5, 0.5, 0.5])
pos1 = np.array([[0.6, 0.5, 0.5]])  # Nearby particle
sigma1 = 1.0
epsilon1 = 1.0
L1 = 10.0
r_c1 = 2.5
assert np.allclose(E_i(r1, pos1, sigma1, epsilon1, L1, r_c1), target)  # Expect some energy value based on interaction: 3999996000000.0083
target = targets[1]

r2 = np.array([1.0, 1.0, 1.0])
pos2 = np.array([[1.5, 1.0, 1.0], [1.5, 1.5, 1.5]])  # One near, one far away
sigma2 = 1.0
epsilon2 = 1.0
L2 = 10.0
r_c2 = 1.5
assert np.allclose(E_i(r2, pos2, sigma2, epsilon2, L2, r_c2), target)  # Expect 16140.993141289438
target = targets[2]

r3 = np.array([0.0, 0.0, 0.0])
pos3 = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])  # All particles are far
sigma3 = 1.0
epsilon3 = 1.0
L3 = 10.0
r_c3 = 2.5
assert np.allclose(E_i(r3, pos3, sigma3, epsilon3, L3, r_c3), target)  # Expect zero energy as no particles are within the cut-off
