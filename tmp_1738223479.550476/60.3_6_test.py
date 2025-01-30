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
    
    # Generate a random position for the test particle within the cubic box using a halton sequence
    def halton_sequence(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f = f / base
            result = result + f * (i % base)
            i = i // base
        return result

    # Generate a deterministic position using the Halton sequence for low-discrepancy
    test_particle_pos = np.array([halton_sequence(np.random.randint(0, 100), 2),
                                  halton_sequence(np.random.randint(0, 100), 3),
                                  halton_sequence(np.random.randint(0, 100), 5)]) * L

    # Function to compute the interaction energy using recursive splitting
    def recursive_energy(test_pos, particles, sigma, epsilon, cutoff, box_length, depth=3):
        if depth == 0 or len(particles) == 0:
            return 0.0
        
        # Calculate minimum image distance with recursive splitting
        delta = particles - test_pos
        delta = delta - box_length * np.round(delta / box_length)
        dist_sq = np.sum(delta**2, axis=1)
        
        # Apply cutoff
        within_cutoff = dist_sq < cutoff**2
        valid_dist_sq = dist_sq[within_cutoff]
        
        # Lennard-Jones potential
        inv_dist2 = (sigma**2) / valid_dist_sq
        inv_dist6 = inv_dist2**3
        inv_dist12 = inv_dist6**2
        energy = np.sum(4 * epsilon * (inv_dist12 - inv_dist6))
        
        # Divide particles into quadrants/octants and recurse
        mid_point = np.median(particles, axis=0)
        less_than_mid = particles < mid_point
        more_than_mid = particles >= mid_point
        
        energy += recursive_energy(test_pos, particles[less_than_mid], sigma, epsilon, cutoff, box_length, depth-1)
        energy += recursive_energy(test_pos, particles[more_than_mid], sigma, epsilon, cutoff, box_length, depth-1)
        
        return energy

    # Calculate the energy change due to the test particle insertion
    energy_change = recursive_energy(test_particle_pos, pos, sigma, epsilon, r_c, L)
    
    # Calculate and return the Boltzmann factor
    boltzmann_factor = np.exp(-beta * energy_change)
    
    return boltzmann_factor


try:
    targets = process_hdf5_to_tuple('60.3', 3)
    target = targets[0]
    pos1 = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    sigma1 = 1.0
    epsilon1 = 3.6
    L1 = 5
    r_c1 = 3
    T1 = 10
    np.random.seed(0)
    assert np.allclose(Widom_insertion(pos1, sigma1, epsilon1, L1, r_c1, T1), target)  # Expected to be 1.0185805629757558

    target = targets[1]
    pos2 = np.array([[2.5, 5.0, 5.0]])  # One particle in the box
    sigma2 = 1.0
    epsilon2 = 1.0
    L2 = 10.0
    r_c2 = 8
    T2 = 10
    np.random.seed(0)
    assert np.allclose(Widom_insertion(pos2, sigma2, epsilon2, L2, r_c2, T2), target)  # Expect to be  1.0000546421925063

    target = targets[2]
    np.random.seed(1)
    L3 = 5
    pos3 = np.random.uniform(0, L3, size=(10, 3))  # Ten particles randomly distributed
    sigma3 = 2.0
    epsilon3 = 0.5
    r_c3 = 3
    T3 = 10
    np.random.seed(0)
    assert np.allclose(Widom_insertion(pos3, sigma3, epsilon3, L3, r_c3, T3), target)  # Expect to be 2.998541562462041e-17

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e