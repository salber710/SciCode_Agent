import numpy as np

# Background: In a lattice model, each site can interact with its nearest neighbors. For a 2D lattice with periodic boundary conditions, 
# the lattice "wraps around" such that the edges are connected. This means that for a site at the edge of the lattice, 
# its neighbors include sites on the opposite edge. For a site (i, j) in an N x N lattice, the neighbors are:
# - Left neighbor: (i, j-1) which wraps to (i, N-1) if j is 0
# - Right neighbor: (i, j+1) which wraps to (i, 0) if j is N-1
# - Above neighbor: (i-1, j) which wraps to (N-1, j) if i is 0
# - Below neighbor: (i+1, j) which wraps to (0, j) if i is N-1

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    # Calculate neighbors with periodic boundary conditions
    left = (i, (j - 1) % N)
    right = (i, (j + 1) % N)
    above = ((i - 1) % N, j)
    below = ((i + 1) % N, j)
    
    return [left, above, right, below]



# Background: In the Ising model, the energy of a particular site in a lattice is determined by its interaction with its nearest neighbors.
# The energy of a site (i, j) is given by the formula: E_ij = -S_ij * (S_left + S_right + S_above + S_below), where S_ij is the spin at site (i, j),
# and S_left, S_right, S_above, S_below are the spins of its nearest neighbors. The negative sign indicates that aligned spins (same sign) lower the energy,
# which is a characteristic of ferromagnetic interactions. The total energy of the site is the sum of these interactions.


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
    # Get the spin of the current site
    S_ij = lattice[i, j]
    
    # Calculate the indices of the neighbors using periodic boundary conditions
    left = lattice[i, (j - 1) % N]
    right = lattice[i, (j + 1) % N]
    above = lattice[(i - 1) % N, j]
    below = lattice[(i + 1) % N, j]
    
    # Calculate the energy contribution from the site (i, j)
    energy = -S_ij * (left + right + above + below)
    
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
