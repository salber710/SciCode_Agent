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
    beta = 1.0 / (k_B * T)
    
    # Randomly generate a new particle position within the box
    r_new = np.random.rand(3) * L
    
    # Calculate the total potential energy of the new particle with existing particles
    E_insert = 0.0
    for r_j in pos:
        # Calculate the distance using the minimum image convention
        delta = r_j - r_new
        delta = delta - L * np.round(delta / L)
        
        # Calculate the squared distance
        r_sq = np.sum(delta**2)
        
        # Check if within cutoff
        if r_sq < r_c**2:
            inv_r6 = (sigma**2 / r_sq)**3
            inv_r12 = inv_r6 * inv_r6
            E_ij = 4 * epsilon * (inv_r12 - inv_r6)
            E_insert += E_ij
    
    # Calculate the Boltzmann factor for the insertion
    Boltz = np.exp(-beta * E_insert)
    
    return Boltz


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