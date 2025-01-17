import numpy as np

# Background: In computational simulations, especially those involving particles in a confined space, 
# it is common to use periodic boundary conditions (PBCs) to simulate an infinite system using a finite 
# simulation box. When a particle moves out of one side of the box, it re-enters from the opposite side. 
# This is akin to the concept of wrapping around in a toroidal space. The goal of applying PBCs is to 
# ensure that the coordinates of particles remain within the bounds of the simulation box. For a cubic 
# box of size L, if a coordinate exceeds L, it should be wrapped around by subtracting L, and if it is 
# less than 0, it should be wrapped by adding L. This ensures that all coordinates are within the range 
# [0, L).


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    # Use numpy's modulus operation to wrap the coordinates
    coord = np.mod(r, L)
    return coord



# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is characterized by two parameters: sigma (σ), which is the distance at which the potential reaches its minimum, and epsilon (ε), 
# which is the depth of the potential well. The potential is given by the formula:
# V(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between two particles.
# In a periodic system, the minimum image convention is used to calculate the shortest distance between particles, 
# considering the periodic boundaries. This ensures that interactions are calculated correctly in a finite simulation box.
# The cut-off distance (r_c) is used to limit the range of interactions, improving computational efficiency by ignoring 
# interactions beyond this distance.


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
        r2 = np.dot(delta, delta)
        
        if r2 < r_c**2:
            # Calculate Lennard-Jones potential
            r6 = (sigma**2 / r2)**3
            r12 = r6**2
            return 4 * epsilon * (r12 - r6)
        else:
            return 0.0

    # Calculate the total energy for particle r
    total_energy = 0.0
    for r_j in pos:
        if not np.array_equal(r, r_j):  # Ensure we do not calculate self-interaction
            total_energy += E_ij(r, r_j, sigma, epsilon, L)

    return total_energy


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
