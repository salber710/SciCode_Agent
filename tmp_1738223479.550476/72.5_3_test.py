from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    
    # Define a custom function to handle periodic boundary conditions using bitwise operations
    def wrap_with_bitwise(x, max_val):
        return x & (max_val - 1)
    
    # Compute neighbors using the custom wrapping function
    neighbors = [
        (wrap_with_bitwise(i - 1 + N, N), j),  # left
        (i, wrap_with_bitwise(j + 1 + N, N)),  # above
        (wrap_with_bitwise(i + 1 + N, N), j),  # right
        (i, wrap_with_bitwise(j - 1 + N, N))   # below
    ]
    
    return neighbors



def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j) using a different approach to handle periodic conditions.'''
    i, j = site

    # Using a dictionary to calculate neighbors with periodic boundary condition
    neighbors = {
        'left': ((i - 1 + N) % N, j),
        'above': (i, (j + 1 + N) % N),
        'right': ((i + 1) % N, j),
        'below': (i, (j - 1 + N) % N)
    }

    return neighbors.values()

def energy_site(i, j, lattice):
    '''Calculate the energy of site (i, j) using numpy broadcasting for neighbor interaction.
    Args:
        i (int): site index along x
        j (int): site index along y
        lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy of site (i, j)
    '''
    N = lattice.shape[0]
    spin = lattice[i, j]

    # Get neighbors using the dictionary values
    neighbors = neighbor_list((i, j), N)

    # Calculate energy using numpy broadcasting and array operations
    neighbor_spins = np.array([lattice[ni, nj] for ni, nj in neighbors])
    energy = -spin * np.dot(neighbor_spins, [1, 1, 1, 1])

    return energy



def energy(lattice):
    '''calculate the total energy for the site (i, j) of the periodic Ising model with dimension (N, N)
    Args: lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy 
    '''
    N = lattice.shape[0]
    total_energy = 0
    
    # Consider diagonal neighbors for variety, even though they don't contribute in the standard model
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            # Calculate energy by considering right and down neighbors
            right_neighbor_energy = spin * lattice[i, (j + 1) % N]
            down_neighbor_energy = spin * lattice[(i + 1) % N, j]
            
            # Add the contributions to the total energy
            total_energy += right_neighbor_energy + down_neighbor_energy
    
    # Return the negative of the accumulated energy to match the typical Ising model convention
    return -total_energy


def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: 
    '''
    mag = 0
    for index, value in np.ndenumerate(spins):
        mag += value
    return float(mag)




def get_flip_probability_magnetization(lattice, i, j, beta):
    '''Calculate spin flip probability and change in total magnetization.
    Args:
        lattice (np.array): shape (N, N), 2D lattice of 1 and -1
        i (int): site index along x
        j (int): site index along y
        beta (float): inverse temperature
    Return:
        A (float): acceptance ratio
        dM (int): change in magnetization after the spin flip
    '''
    N = lattice.shape[0]
    spin = lattice[i, j]

    # Define neighbors with explicit boundary wrapping
    neighbor_indices = [
        ((i - 1) % N, j),  # left
        ((i + 1) % N, j),  # right
        (i, (j - 1) % N),  # above
        (i, (j + 1) % N)   # below
    ]

    # Calculate ΔE with a different mathematical approach to summing
    delta_E = 0
    for ni, nj in neighbor_indices:
        delta_E += lattice[ni, nj]
    delta_E *= 2 * spin

    # Use an alternative probability formula based on logistic function
    A = 1 / (1 + np.exp(beta * delta_E))

    # Magnetization change remains the same
    dM = -2 * spin

    return A, dM


try:
    targets = process_hdf5_to_tuple('72.5', 4)
    target = targets[0]
    lattice = np.array([[ 1, -1,  1,  1],[-1, -1,  1,  1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
    assert np.allclose(get_flip_probability_magnetization(lattice, 1, 2, 1), target)

    target = targets[1]
    lattice = np.array([[ 1, -1,  1,  1],[-1, -1,  -1,  1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
    assert np.allclose(get_flip_probability_magnetization(lattice, 1, 2, 1), target)

    target = targets[2]
    lattice = np.array([[ 1, -1,  1,  1],[-1, -1,  1,  -1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
    assert np.allclose(get_flip_probability_magnetization(lattice, 1, 2, 1), target)

    target = targets[3]
    def test_spin_flip():
        params = {
            'i': 1, 'j': 2,
            'lattice': np.array([
                [ 1, -1,  1,  1],
                [-1, -1,  1,  1],
                [-1, -1,  1,  1],
                [ 1, -1, -1, -1]
            ]),
            'beta': 1
        }
        return get_flip_probability_magnetization(**params) == (0.01831563888873418, -2)
    assert test_spin_flip() == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e