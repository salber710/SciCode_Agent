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

    # Precompute the neighbors for efficiency
    neighbors = np.zeros((N, N, 2), dtype=int)

    # Fill neighbors array with indices of right and down neighbors
    for i in range(N):
        for j in range(N):
            neighbors[i, j, 0] = lattice[i, (j + 1) % N]  # right neighbor
            neighbors[i, j, 1] = lattice[(i + 1) % N, j]  # down neighbor

    # Compute energy using precomputed neighbors
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            right_neighbor = neighbors[i, j, 0]
            down_neighbor = neighbors[i, j, 1]

            # Subtract the interaction energies directly
            total_energy -= spin * (right_neighbor + down_neighbor)

    return total_energy


try:
    targets = process_hdf5_to_tuple('72.3', 4)
    target = targets[0]
    lattice = np.array([[1, 1, 1, -1],[-1, 1, -1, -1],[-1, -1, 1, 1],[-1, 1, 1, 1]])
    assert np.allclose(energy(lattice), target)

    target = targets[1]
    lattice = np.array([[1, 1, 1, -1],[-1, -1, -1, -1],[-1, -1, 1, 1],[-1, 1, 1, 1]])
    assert np.allclose(energy(lattice), target)

    target = targets[2]
    lattice = np.array([[1, 1, 1, -1],[-1, 1, -1, 1],[-1, -1, 1, 1],[-1, 1, 1, 1]])
    assert np.allclose(energy(lattice), target)

    target = targets[3]
    def test_energy():
        params = {
            'lattice': np.array([
                [1, 1, 1, -1],
                [-1, 1, -1, -1],
                [-1, -1, 1, 1],
                [-1, 1, 1, 1]
            ])
        }
        return energy(**params) == 0
    assert test_energy() == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e