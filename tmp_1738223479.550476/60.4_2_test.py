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
    
    # Determine the number of divisions per side, taking the smallest integer greater than or equal to cube root of N
    div = int(np.ceil(N ** (1/3)))
    
    # Calculate the spacing between particles
    spacing = L / div
    
    # Create a meshgrid for particle positions
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(div) * spacing,
        np.arange(div) * spacing,
        np.arange(div) * spacing
    )
    
    # Flatten the grid arrays and stack them into a single array of coordinates
    positions = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
    
    # Trim the array to contain exactly N particles
    positions = positions[:N]
    
    return positions, L


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e