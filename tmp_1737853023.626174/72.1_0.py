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
    i, j = site
    # Calculate neighbors with periodic boundary conditions
    i_above = (i - 1) % N
    j_right = (j + 1) % N
    i_below = (i + 1) % N
    j_left = (j - 1) % N
    
    # Return the list of neighbors
    return [(i, j_left), (i_above, j), (i, j_right), (i_below, j)]

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('72.1', 4)
target = targets[0]

assert np.allclose(neighbor_list((0, 0), 10), target)
target = targets[1]

assert np.allclose(neighbor_list((9, 9), 10), target)
target = targets[2]

assert np.allclose(neighbor_list((0, 5), 10), target)
target = targets[3]

def test_neighbor():
    N = 10
    inputs = [(0, 0), (9, 9), (0, 5)]
    corrects = [
        [(9, 0), (0, 1), (1, 0), (0, 9)],
        [(8, 9), (9, 0), (0, 9), (9, 8)],
        [(9, 5), (0, 6), (1, 5), (0, 4)]
    ]
    for (i, j), correct in zip(inputs, corrects):
        if neighbor_list((i, j), N) != correct:
            return False
    return True
assert (test_neighbor()) == target
