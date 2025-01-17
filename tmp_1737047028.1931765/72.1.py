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
