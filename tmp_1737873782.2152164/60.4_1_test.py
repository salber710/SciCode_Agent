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

    # Determine the number of particles per side in the grid
    n_side = int(np.ceil(N**(1/3)))

    # Calculate the spacing between particles in the grid
    spacing = L / n_side

    # Initialize an array to hold the particle positions
    positions = []

    # Arrange particles in a cubic grid
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                if len(positions) < N:
                    positions.append([x * spacing, y * spacing, z * spacing])
    
    # Convert positions list to a numpy array
    positions = np.array(positions)
    
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