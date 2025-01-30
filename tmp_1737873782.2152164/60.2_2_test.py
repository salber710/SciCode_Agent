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

    def minimum_image(r_i, r_j, L):
        '''Apply the minimum image convention to find the shortest distance vector between two particles.'''
        d = r_j - r_i
        d -= L * np.round(d / L)
        return d

    def E_ij(r_ij, sigma, epsilon, r_c):
        '''Calculate the Lennard-Jones potential energy between two particles at distance r_ij.'''
        r2 = np.dot(r_ij, r_ij)
        if r2 < r_c ** 2:
            r2_inv = sigma ** 2 / r2
            r6_inv = r2_inv ** 3
            r12_inv = r6_inv ** 2
            energy = 4 * epsilon * (r12_inv - r6_inv)
            return energy
        else:
            return 0.0

    total_energy = 0.0
    for other_pos in pos:
        if not np.array_equal(r, other_pos):  # Ensure we don't calculate self-interaction
            r_ij = minimum_image(r, other_pos, L)
            total_energy += E_ij(r_ij, sigma, epsilon, r_c)

    return total_energy


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e