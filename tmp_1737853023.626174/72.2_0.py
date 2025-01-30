import numpy as np

# Background: In a lattice model, each site has a set of nearest neighbors. For a 2D lattice with periodic boundary conditions, 
# the neighbors of a site (i, j) are determined by considering the lattice as a torus. This means that if a site is at the edge 
# of the lattice, its neighbors wrap around to the opposite edge. For a site (i, j) in an N x N lattice, the neighbors are:
# (i-1, j) - the site above, (i, j+1) - the site to the right, (i+1, j) - the site below, and (i, j-1) - the site to the left.
# The periodic boundary conditions ensure that these indices wrap around using modulo N arithmetic.

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    if not isinstance(N, int):
        raise TypeError("N must be an integer")
    if not isinstance(site, tuple) or not len(site) == 2 or not all(isinstance(x, int) for x in site):
        raise TypeError("site must be a tuple of two integers")
    if any(x >= N or x < 0 for x in site):
        raise IndexError("site indices must be within the range of the lattice size")
    if N <= 0:
        raise IndexError("N must be a positive integer")

    i, j = site
    # Calculate neighbors with periodic boundary conditions
    i_above = (i - 1) % N
    j_right = (j + 1) % N
    i_below = (i + 1) % N
    j_left = (j - 1) % N
    
    # Return the list of neighbors
    return [(i, j_left), (i_above, j), (i, j_right), (i_below, j)]



# Background: In the Ising model, the energy of a particular site in a lattice is determined by its interaction with its nearest neighbors.
# The energy contribution of a site (i, j) is calculated using the formula: E_ij = -S_ij * (S_neighbors_sum), where S_ij is the spin at site (i, j),
# and S_neighbors_sum is the sum of the spins of its nearest neighbors. The negative sign indicates that aligned spins (same sign) lower the energy,
# which is a characteristic of ferromagnetic interactions. The periodic boundary conditions ensure that the lattice behaves like a torus, 
# meaning that spins on the edges interact with those on the opposite edges.


def energy_site(i, j, lattice):
    '''Calculate the energy of site (i, j)
    Args:
        i (int): site index along x
        j (int): site index along y
        lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy of site (i, j)
    '''
    N = lattice.shape[0]
    
    # Get the spin at the current site
    S_ij = lattice[i, j]
    
    # Calculate the indices of the neighbors using periodic boundary conditions
    i_above = (i - 1) % N
    j_right = (j + 1) % N
    i_below = (i + 1) % N
    j_left = (j - 1) % N
    
    # Sum the spins of the nearest neighbors
    S_neighbors_sum = (lattice[i_above, j] + lattice[i, j_right] +
                       lattice[i_below, j] + lattice[i, j_left])
    
    # Calculate the energy of the site
    energy = -S_ij * S_neighbors_sum
    
    return energy

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('72.2', 4)
target = targets[0]

i = 1
j = 2
lattice = np.array([[ 1, -1,  1,  1],[-1, -1,  1,  1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
assert np.allclose(energy_site(i, j, lattice), target)
target = targets[1]

i = 1
j = 2
lattice = np.array([[ 1, -1,  1,  1],[-1, 1,  1,  1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
assert np.allclose(energy_site(i, j, lattice), target)
target = targets[2]

i = 1
j = 2
lattice = np.array([[ 1, -1,  1,  1],[-1, -1,  1,  -1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
assert np.allclose(energy_site(i, j, lattice), target)
target = targets[3]

def test_energy_site():
    params = {
        'i': 1, 'j': 2,
        'lattice': np.array([
            [ 1, -1,  1,  1],
            [-1, -1,  1,  1],
            [-1, -1,  1,  1],
            [ 1, -1, -1, -1]
        ])
    }
    return energy_site(**params) == -1*(-1 + 3)
assert test_energy_site() == target
