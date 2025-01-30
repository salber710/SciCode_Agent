from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    coord = np.mod(r, L)
    return coord



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
        '''Calculate the Lennard-Jones potential between two particles with periodic boundary conditions.
        Parameters:
        r_i, r_j : array-like, the (x, y, z) coordinates of the two particles.
        sigma : float, the distance at which the potential minimum occurs.
        epsilon : float, the depth of the potential well.
        L : float, the length of the side of the cubic box.
        r_c : float, cut-off distance.
        Returns:
        float, the Lennard-Jones potential energy between the two particles.
        '''
        # Apply minimum image convention
        delta = r_j - r_i
        delta = delta - L * np.round(delta / L)
        
        # Calculate the distance
        r_sq = np.sum(delta**2)
        
        # Check if within cutoff
        if r_sq < r_c**2:
            inv_r6 = (sigma**2 / r_sq)**3
            inv_r12 = inv_r6 * inv_r6
            return 4 * epsilon * (inv_r12 - inv_r6)
        else:
            return 0.0
    
    # Total energy of the particle due to interactions
    E = 0.0
    for r_j in pos:
        if not np.array_equal(r, r_j):  # Avoid self-interaction
            E += E_ij(r, r_j, sigma, epsilon, L, r_c)
    
    return E



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
    k_B = 1.0  # Boltzmann constant in reduced units
    beta = 1.0 / (k_B * T)  # Inverse thermal energy

    # Randomly insert a particle into the simulation box
    r_test = np.random.rand(3) * L
    
    # Calculate the energy of the test particle due to its interaction with all others
    E_insert = 0.0
    for r_j in pos:
        # Apply minimum image convention for periodic boundary conditions
        delta = r_test - r_j
        delta = delta - L * np.round(delta / L)
        
        r_sq = np.sum(delta**2)
        
        # Calculate interaction energy if within cutoff
        if r_sq < r_c**2:
            inv_r6 = (sigma**2 / r_sq)**3
            inv_r12 = inv_r6 * inv_r6
            E_insert += 4 * epsilon * (inv_r12 - inv_r6)

    # Calculate the Boltzmann factor for the test particle
    Boltz = np.exp(-beta * E_insert)
    
    return Boltz



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
    
    # Determine the number of particles per side for the grid
    n_side = int(np.ceil(N**(1/3)))
    
    # Create a grid of positions
    positions = []
    spacing = L / n_side  # Distance between particles in the grid
    
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                if len(positions) < N:
                    positions.append([x * spacing, y * spacing, z * spacing])
    
    positions = np.array(positions[:N])
    
    return positions, L



def MC(positions, sigma, epsilon, L, r_c, T, n_eq, n_prod, insertion_freq, move_magnitude):
    '''Perform Monte Carlo simulations using the Metropolis-Hastings algorithm and Widom insertion method to calculate system energies and chemical potential.
    Parameters:
    positions : ndarray
        An array of (x, y, z) coordinates for every particle.
    sigma, epsilon : float
        Parameters of the Lennard-Jones potential.
    L : float
        The length of each side of the cubic simulation box.
    r_c : float
        Cutoff radius beyond which the LJ potential is considered zero.
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

    k_B = 1.0  # Boltzmann constant in reduced units
    beta = 1.0 / (k_B * T)  # Inverse thermal energy
    N = len(positions)

    def E_i(r, pos, sigma, epsilon, L, r_c):
        '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.'''
        def E_ij(r_i, r_j, sigma, epsilon, L, r_c):
            '''Calculate the Lennard-Jones potential between two particles with periodic boundary conditions.'''
            delta = r_j - r_i
            delta = delta - L * np.round(delta / L)
            r_sq = np.sum(delta**2)
            if r_sq < r_c**2:
                inv_r6 = (sigma**2 / r_sq)**3
                inv_r12 = inv_r6 * inv_r6
                return 4 * epsilon * (inv_r12 - inv_r6)
            else:
                return 0.0
        
        E = 0.0
        for r_j in pos:
            if not np.array_equal(r, r_j):
                E += E_ij(r, r_j, sigma, epsilon, L, r_c)
        return E

    def Widom_insertion(pos, sigma, epsilon, L, r_c, T):
        '''Perform the Widom test particle insertion method to calculate the change in chemical potential.'''
        r_test = np.random.rand(3) * L
        E_insert = 0.0
        for r_j in pos:
            delta = r_test - r_j
            delta = delta - L * np.round(delta / L)
            r_sq = np.sum(delta**2)
            if r_sq < r_c**2:
                inv_r6 = (sigma**2 / r_sq)**3
                inv_r12 = inv_r6 * inv_r6
                E_insert += 4 * epsilon * (inv_r12 - inv_r6)
        Boltz = np.exp(-beta * E_insert)
        return Boltz

    # Initialize simulation
    n_accp = 0
    energies = []
    chemical_potentials = []

    # Equilibration phase
    for _ in range(n_eq):
        for i in range(N):
            current_pos = positions[i]
            current_energy = E_i(current_pos, positions, sigma, epsilon, L, r_c)

            # Propose a new position
            new_pos = current_pos + (np.random.rand(3) - 0.5) * move_magnitude
            new_pos = np.mod(new_pos, L)  # Apply periodic boundary conditions

            # Calculate new energy
            new_energy = E_i(new_pos, positions, sigma, epsilon, L, r_c)

            # Metropolis criterion
            if np.random.rand() < np.exp(-beta * (new_energy - current_energy)):
                positions[i] = new_pos
                n_accp += 1

    # Production phase
    for step in range(n_prod):
        for i in range(N):
            current_pos = positions[i]
            current_energy = E_i(current_pos, positions, sigma, epsilon, L, r_c)

            # Propose a new position
            new_pos = current_pos + (np.random.rand(3) - 0.5) * move_magnitude
            new_pos = np.mod(new_pos, L)  # Apply periodic boundary conditions

            # Calculate new energy
            new_energy = E_i(new_pos, positions, sigma, epsilon, L, r_c)

            # Metropolis criterion
            if np.random.rand() < np.exp(-beta * (new_energy - current_energy)):
                positions[i] = new_pos
                n_accp += 1

        # Collect energy data
        total_energy = sum(E_i(pos, positions, sigma, epsilon, L, r_c) for pos in positions)
        energies.append(total_energy)

        # Perform Widom insertion at specified frequency
        if step % insertion_freq == 0:
            widom_factor = Widom_insertion(positions, sigma, epsilon, L, r_c, T)
            chemical_potentials.append(-np.log(widom_factor) / beta)

    accp_ratio = n_accp / (n_eq + n_prod) / N

    return np.array(energies), np.array(chemical_potentials), n_accp, accp_ratio


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e