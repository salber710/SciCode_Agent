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

    def E_ij(r_i, r_j, sigma, epsilon, L, r_c):
        '''Calculate the Lennard-Jones potential energy between two particles using minimum image convention.'''
        # Calculate the distance vector considering periodic boundaries
        delta = r_j - r_i
        delta = delta - L * np.round(delta / L)  # Minimum image convention
        # Calculate the distance
        r_ij = np.linalg.norm(delta)
        # Calculate the Lennard-Jones potential if within the cut-off distance
        if r_ij < r_c and r_ij != 0:
            sr6 = (sigma / r_ij) ** 6
            sr12 = sr6 ** 2
            return 4 * epsilon * (sr12 - sr6)
        else:
            return 0.0

    # Initialize total energy
    E = 0.0
    # Calculate the total energy by summing the pairwise interactions
    for r_j in pos:
        if not np.array_equal(r, r_j):  # Avoid self-interaction
            E += E_ij(r, r_j, sigma, epsilon, L, r_c)

    return E


# Background: The Widom test particle insertion method is a technique used in statistical mechanics to estimate the chemical potential of a system. 
# In the context of a Lennard-Jones system in the NVT ensemble (constant number of particles, volume, and temperature), this method involves 
# inserting a "test" particle into the system and calculating the change in energy due to this insertion. The chemical potential is related to 
# the probability of successfully inserting a particle without significantly altering the system's energy. The Boltzmann factor, e^(-beta * ΔE), 
# where ΔE is the change in energy and beta is the inverse temperature (1/kT, with k being the Boltzmann constant), quantifies this probability. 
# The average of this factor over many insertions gives an estimate of the chemical potential.


def Widom_insertion(pos, sigma, epsilon, L, r_c, T):
    '''Perform the Widom test particle insertion method to calculate the change in chemical potential.
    Parameters:
    pos : ndarray, Array of position vectors of all particles in the system.
    sigma: float, The effective particle diameter 
    epsilon: float, The depth of the potential well
    L: float, The length of each side of the cubic simulation box
    r_c: float, Cut-Off Distance
    T: float, The temperature of the system
    Returns:
    float: Boltzmann factor for the test particle insertion, e^(-beta * energy of insertion).
    '''
    # Boltzmann constant
    k_B = 1.0  # Assuming units where k_B = 1 for simplicity
    beta = 1.0 / (k_B * T)
    
    # Randomly generate a position for the test particle within the box
    r_test = np.random.uniform(0, L, size=3)
    
    # Calculate the energy change due to the insertion of the test particle
    def E_ij(r_i, r_j, sigma, epsilon, L, r_c):
        '''Calculate the Lennard-Jones potential energy between two particles using minimum image convention.'''
        delta = r_j - r_i
        delta = delta - L * np.round(delta / L)  # Minimum image convention
        r_ij = np.linalg.norm(delta)
        if r_ij < r_c:
            sr6 = (sigma / r_ij) ** 6
            sr12 = sr6 ** 2
            return 4 * epsilon * (sr12 - sr6)
        else:
            return 0.0

    # Calculate the total energy change due to the test particle
    delta_E = 0.0
    for r_j in pos:
        delta_E += E_ij(r_test, r_j, sigma, epsilon, L, r_c)
    
    # Calculate the Boltzmann factor
    Boltz = np.exp(-beta * delta_E)
    
    # Ensure the Boltzmann factor does not exceed 1 due to numerical precision issues
    return min(Boltz, 1.0)



# Background: In molecular dynamics simulations, initializing the system involves placing particles in a simulation box
# with a specified density. The density (rho) is defined as the number of particles per unit volume. For a cubic box,
# the volume is L^3, where L is the side length of the cube. Given the number of particles (N) and the density (rho),
# the side length L can be calculated as L = (N / rho)^(1/3). Once L is determined, particles can be arranged in a
# regular grid within the box. This ensures that particles are evenly distributed and do not overlap, providing a
# suitable starting configuration for simulations.


def init_system(N, rho):
    '''Initialize a system of particles arranged on a cubic grid within a cubic box.
    Args:
    N (int): The number of particles to be placed in the box.
    rho (float): The density of particles within the box, defined as the number of particles per unit volume.
    Returns:
    tuple: A tuple containing:
        - positions(np.ndarray): The array of particle positions in a 3D space.
        - L(float): The length of the side of the cubic box in which the particles are placed.
    '''
    # Calculate the side length of the cubic box
    L = (N / rho) ** (1/3)
    
    # Determine the number of particles per side of the grid
    n_side = int(np.ceil(N ** (1/3)))
    
    # Create a grid of points within the box
    grid_spacing = L / n_side
    positions = []
    
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                if len(positions) < N:
                    positions.append([x * grid_spacing, y * grid_spacing, z * grid_spacing])
    
    # Convert positions to a numpy array
    positions = np.array(positions)
    
    return positions, L

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('60.4', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
N1 = 8  # Number of particles
rho1 = 1  # Density
assert cmp_tuple_or_list(init_system(N1, rho1), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
N2 = 10
rho2 = 1
positions2, L2 = init_system(N2, rho2)
assert cmp_tuple_or_list((positions2[:10], L2, len(positions2)), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
N3 = 27  # Cube of 3
rho3 = 27  # Very high density (particle per unit volume)
assert cmp_tuple_or_list(init_system(N3, rho3), target)
