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


# Background: The Widom test particle insertion method is a technique used in statistical mechanics to estimate the 
# chemical potential of a system. In the context of a Lennard-Jones system in the NVT ensemble, the method involves 
# inserting a "test" particle into the system and calculating the change in energy due to this insertion. The chemical 
# potential is related to the probability of successfully inserting a particle without significantly altering the system's 
# energy. The Boltzmann factor, e^(-beta * ΔE), where ΔE is the change in energy upon insertion and beta is the inverse 
# temperature (1/kT), is used to quantify this probability. The average of this factor over many insertions gives an 
# estimate of the chemical potential.


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
    k_B = 1.0  # Assuming units where k_B = 1

    # Inverse temperature
    beta = 1.0 / (k_B * T)

    # Randomly generate a position for the test particle within the box
    r_test = np.random.uniform(0, L, 3)

    # Calculate the energy change due to the insertion of the test particle
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

    # Calculate the total energy change for inserting the test particle
    delta_E = 0.0
    for r_j in pos:
        delta_E += E_ij(r_test, r_j, sigma, epsilon, L)

    # Calculate the Boltzmann factor
    Boltz = np.exp(-beta * delta_E)

    return Boltz


# Background: In molecular simulations, initializing the system involves placing particles in a simulation box
# with a specified density. The density (rho) is defined as the number of particles per unit volume. For a 
# cubic box, the volume is L^3, where L is the side length of the cube. Given the number of particles (N) 
# and the density (rho), the side length L can be calculated as L = (N / rho)^(1/3). Once L is determined, 
# particles can be arranged in a regular grid within the cube. This ensures that particles are evenly 
# distributed and do not overlap, providing a good starting configuration for simulations.

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

    # Determine the number of particles per side of the cube
    n_side = int(np.ceil(N ** (1/3)))

    # Create a grid of positions
    positions = []
    spacing = L / n_side  # Distance between particles in the grid

    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                if len(positions) < N:
                    positions.append([x * spacing, y * spacing, z * spacing])

    # Convert positions to a numpy array
    positions = np.array(positions)

    return positions, L



# Background: Monte Carlo simulations using the Metropolis-Hastings algorithm are a powerful tool for sampling 
# the configuration space of a system. In the context of molecular simulations, this involves proposing random 
# displacements of particles and accepting or rejecting these moves based on the change in energy, ΔE, using 
# the Metropolis criterion: accept the move with probability min(1, exp(-βΔE)), where β = 1/(k_B * T) is the 
# inverse temperature. The Widom test particle insertion method is used to estimate the chemical potential by 
# inserting a test particle and calculating the energy change. The simulation consists of equilibration steps 
# to allow the system to reach a steady state, followed by production steps where data is collected. The 
# acceptance ratio of proposed moves is an important metric for assessing the efficiency of the simulation.


def MC(N, sigma, epsilon, r_c, rho, T, n_eq, n_prod, insertion_freq, move_magnitude):
    '''Perform Monte Carlo simulations using the Metropolis-Hastings algorithm and Widom insertion method to calculate system energies and chemical potential.
    Parameters:
    N (int): The number of particles to be placed in the box.
    sigma, epsilon : float
        Parameters of the Lennard-Jones potential.
    r_c : float
        Cutoff radius beyond which the LJ potential is considered zero.
    rho (float): The density of particles within the box, defined as the number of particles per unit volume.
    T : float
        Temperature of the system.
    n_eq : int
        Number of equilibration steps in the simulation.
    n_prod : int
        Number of production steps in the simulation.
    insertion_freq : int
        Frequency of performing Widom test particle insertions after equilibration.
    move_magnitude : float
        Magnitude of the random displacement in particle movement.
    Returns:
    tuple
        Returns a tuple containing the corrected energy array, extended chemical potential,
        number of accepted moves, and acceptance ratio.
    '''
    # Initialize the system
    positions, L = init_system(N, rho)
    
    # Boltzmann constant
    k_B = 1.0  # Assuming units where k_B = 1

    # Inverse temperature
    beta = 1.0 / (k_B * T)

    # Initialize variables for tracking
    n_accp = 0
    E = []
    ecp = []

    # Function to calculate the energy of a single particle
    def E_i(r, pos, sigma, epsilon, L, r_c):
        '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.'''
        def E_ij(r_i, r_j, sigma, epsilon, L):
            '''Calculate the Lennard-Jones potential energy between two particles using minimum image convention.'''
            delta = r_j - r_i
            delta = delta - L * np.round(delta / L)  # Minimum image convention
            r2 = np.dot(delta, delta)
            if r2 < r_c**2:
                r6 = (sigma**2 / r2)**3
                r12 = r6**2
                return 4 * epsilon * (r12 - r6)
            else:
                return 0.0

        total_energy = 0.0
        for r_j in pos:
            if not np.array_equal(r, r_j):
                total_energy += E_ij(r, r_j, sigma, epsilon, L)
        return total_energy

    # Equilibration phase
    for step in range(n_eq):
        for i in range(N):
            # Propose a random move
            old_pos = positions[i].copy()
            new_pos = old_pos + np.random.uniform(-move_magnitude, move_magnitude, 3)
            new_pos = wrap(new_pos, L)

            # Calculate energy change
            delta_E = E_i(new_pos, positions, sigma, epsilon, L, r_c) - E_i(old_pos, positions, sigma, epsilon, L, r_c)

            # Metropolis criterion
            if np.random.rand() < np.exp(-beta * delta_E):
                positions[i] = new_pos
                n_accp += 1

    # Production phase
    for step in range(n_prod):
        for i in range(N):
            # Propose a random move
            old_pos = positions[i].copy()
            new_pos = old_pos + np.random.uniform(-move_magnitude, move_magnitude, 3)
            new_pos = wrap(new_pos, L)

            # Calculate energy change
            delta_E = E_i(new_pos, positions, sigma, epsilon, L, r_c) - E_i(old_pos, positions, sigma, epsilon, L, r_c)

            # Metropolis criterion
            if np.random.rand() < np.exp(-beta * delta_E):
                positions[i] = new_pos
                n_accp += 1

        # Calculate total energy of the system
        total_energy = sum(E_i(pos, positions, sigma, epsilon, L, r_c) for pos in positions)
        E.append(total_energy)

        # Perform Widom insertion
        if step % insertion_freq == 0:
            boltzmann_factor = Widom_insertion(positions, sigma, epsilon, L, r_c, T)
            ecp.append(boltzmann_factor)

    # Calculate acceptance ratio
    accp_ratio = n_accp / (n_eq + n_prod) / N

    return E, ecp, n_accp, accp_ratio


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('60.5', 2)
target = targets[0]

epsilon,sigma = 0.0 ,1.0
T = 3.0
r_c = 2.5
N = 216
rho_list = np.arange(0.01,0.9,0.1)
mu_ext_list = np.zeros(len(rho_list))
checks = []
for i in range(len(rho_list)):
    rho = rho_list[i]
    np.random.seed(i)
    E_array,mu_ext, n_accp, accp_rate = MC(N,sigma,epsilon,r_c,rho,T,n_eq = int(1e4), n_prod = int(4e4),
                                            insertion_freq = 2,
                                            move_magnitude = 0.3)
    mu_ext_list[i] = mu_ext
    ## checks.append(np.abs(mu - mu_expected)/mu_expected < 0.05)  ##
    #print("Finish with acceptance rate ", accp_rate)
mu_ext_list = np.array(mu_ext_list)
assert (np.mean(mu_ext_list) == 0) == target
target = targets[1]

epsilon,sigma = 1.0 ,1.0
T = 3.0
r_c = 2.5
N = 216
rho_list = np.arange(0.3,0.9,0.1)
mu_ext_list = np.zeros(len(rho_list))
checks = []
for i in range(len(rho_list)):
    rho = rho_list[i]
    np.random.seed(i**2+1024)
    E_array,mu_ext, n_accp, accp_rate = MC(N,sigma,epsilon,r_c,rho,T,n_eq = int(1e4), n_prod = int(4e4),
                                            insertion_freq = 2,
                                            move_magnitude = 0.3)
    mu_ext_list[i] = mu_ext
    ## checks.append(np.abs(mu - mu_expected)/mu_expected < 0.05)  ##
    #print("Finish with acceptance rate ", accp_rate)
mu_ext_list = np.array(mu_ext_list)
ref = np.array([ 0.39290198,  1.01133745,  2.21399804,  3.70707519,  6.93916947,
       16.13690354, 54.55808743])
assert (np.abs(np.mean((mu_ext_list-ref)/ref)) < 0.1) == target
