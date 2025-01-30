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

    # We will use a different approach by explicitly calculating individual neighbor contributions
    for i in range(N):
        for j in range(N):
            # Calculate energy contribution for each site by considering all four neighbors separately
            spin = lattice[i, j]

            left_neighbor = lattice[i, (j - 1) % N]
            right_neighbor = lattice[i, (j + 1) % N]
            up_neighbor = lattice[(i - 1) % N, j]
            down_neighbor = lattice[(i + 1) % N, j]

            # Add contributions from each neighbor interaction
            total_energy -= spin * left_neighbor
            total_energy -= spin * right_neighbor
            total_energy -= spin * up_neighbor
            total_energy -= spin * down_neighbor

    # Since each pair is counted twice, divide the total energy by 2
    return total_energy / 2


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