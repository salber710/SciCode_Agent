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
    # Use a direct approach with map and lambda to ensure r stays within [0, L)
    return list(map(lambda x: x - L * (x // L) if x >= 0 else x - L * ((x // L) - 1), r))


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

    def E_ij(r1, r2):
        '''Subfunction to calculate the Lennard-Jones potential between two particles.'''
        # Calculate the minimum image distance vector using numpy's vector operations
        delta_r = [(r1[i] - r2[i] + 0.5 * L) % L - 0.5 * L for i in range(3)]
        # Compute the squared distance
        dist_sq = sum(d**2 for d in delta_r)
        # Calculate the Lennard-Jones potential if within cutoff
        if dist_sq < r_c ** 2:
            inv_dist2 = sigma ** 2 / dist_sq
            inv_dist6 = inv_dist2 ** 3
            inv_dist12 = inv_dist6 ** 2
            return 4 * epsilon * (inv_dist12 - inv_dist6)
        else:
            return 0.0

    # Total energy initialization
    total_energy = 0.0
    # Iterate over all other particles using reversed order to calculate the interaction energy
    for other_pos in reversed(pos):
        total_energy += E_ij(r, other_pos)

    return total_energy



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
    
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    beta = 1.0 / (k_B * T)
    
    # Generate a random position for the test particle using Latin Hypercube Sampling

    sampler = LatinHypercube(d=3)
    sample = sampler.random(n=1)
    test_particle_pos = sample[0] * L

    # Function to compute interaction energy using numpy broadcasting
    def compute_energy_broadcast(test_pos, particles, sigma, epsilon, cutoff, box_length):
        delta = particles - test_pos
        delta -= box_length * np.round(delta / box_length)
        dist_sq = np.sum(delta**2, axis=1)
        
        # Apply cutoff
        valid = dist_sq < cutoff**2
        valid_dist_sq = dist_sq[valid]
        
        # Lennard-Jones potential
        inv_dist2 = (sigma**2) / valid_dist_sq
        inv_dist6 = inv_dist2**3
        inv_dist12 = inv_dist6**2
        energy = np.sum(4 * epsilon * (inv_dist12 - inv_dist6))
        
        return energy

    # Calculate the energy change due to the test particle insertion
    energy_change = compute_energy_broadcast(test_particle_pos, pos, sigma, epsilon, r_c, L)
    
    # Calculate and return the Boltzmann factor
    boltzmann_factor = np.exp(-beta * energy_change)
    
    return boltzmann_factor



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
    
    # Calculate the number of particles per side
    num_per_side = int(np.ceil(N ** (1/3)))
    
    # Calculate the spacing between particles
    spacing = L / num_per_side
    
    # Initialize positions list
    positions = []
    
    # Use a spiral pattern for particle placement
    for x in range(num_per_side):
        for y in range(num_per_side):
            if (x + y) % 2 == 0:
                z_range = range(num_per_side)
            else:
                z_range = range(num_per_side - 1, -1, -1)
            for z in z_range:
                if len(positions) < N:
                    offset = 0.5 * (x % 2) * spacing
                    positions.append([
                        (x + 0.5) * spacing + offset,
                        (y + 0.5) * spacing + offset,
                        (z + 0.5) * spacing
                    ])

    # Convert to numpy array and trim to N particles
    positions = np.array(positions[:N])
    
    return positions, L



# Background: The Metropolis-Hastings algorithm is a Monte Carlo method used in statistical physics to sample from a probability distribution. 
# It is particularly useful for simulating the canonical ensemble in which the number of particles, volume, and temperature are fixed. 
# The algorithm involves proposing random moves for particles and accepting or rejecting these moves based on the change in energy and a probability criterion.
# The Widom insertion method is used to calculate the chemical potential by inserting a "test" particle into the system and measuring the resulting energy change.
# In this function, we combine both methods to perform simulations of a Lennard-Jones system. The energy of the system is calculated using the Lennard-Jones potential, 
# and periodic boundary conditions are applied to maintain particles within the simulation box.


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
    k_B = 1.380649e-23
    beta = 1.0 / (k_B * T)
    
    # Prepare to collect data
    energies = []
    chemical_potentials = []
    n_accp = 0
    
    def compute_total_energy(positions):
        total_energy = 0.0
        for i, pos_i in enumerate(positions):
            total_energy += E_i(pos_i, np.delete(positions, i, axis=0), sigma, epsilon, L, r_c)
        return total_energy / 2.0  # Each pair is counted twice
    
    # Start with equilibration
    for step in range(n_eq + n_prod):
        for i in range(N):
            # Randomly select a particle and attempt a move
            current_pos = positions[i]
            new_pos = current_pos + move_magnitude * (np.random.rand(3) - 0.5)
            new_pos = wrap(new_pos, L)  # Apply periodic boundary conditions

            # Calculate energies
            E_old = E_i(current_pos, np.delete(positions, i, axis=0), sigma, epsilon, L, r_c)
            E_new = E_i(new_pos, np.delete(positions, i, axis=0), sigma, epsilon, L, r_c)

            # Metropolis criterion
            delta_E = E_new - E_old
            if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                positions[i] = new_pos
                if step >= n_eq:
                    n_accp += 1

        if step >= n_eq:
            # Record energy
            total_energy = compute_total_energy(positions)
            energies.append(total_energy)

            # Perform Widom insertion if needed
            if (step - n_eq) % insertion_freq == 0:
                boltzmann_factor = Widom_insertion(positions, sigma, epsilon, L, r_c, T)
                chemical_potentials.append(-np.log(np.mean(boltzmann_factor)) / beta)

    # Calculate acceptance ratio
    accp_ratio = n_accp / (n_prod * N)

    # Return results
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