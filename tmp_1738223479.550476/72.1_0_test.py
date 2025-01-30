from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: In physics and computational modeling, periodic boundary conditions (PBC) are used to simulate a small part of a system that behaves as if it's part of an infinite system. This is done by "wrapping" the edges of the system, so that particles exiting one side of the system re-enter from the opposite side. In a 2D lattice, each spin site has four nearest neighbors. For a site at position (i, j), these neighbors are located at (i - 1, j), (i, j + 1), (i + 1, j), and (i, j - 1). When implementing PBC in a lattice of dimension (N, N), indices must wrap around such that if an index goes below 0, it wraps around to N-1, and if it goes above N-1, it wraps around to 0.

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    # Calculate each neighbor with periodic boundary conditions
    i_left = (i - 1) % N
    i_right = (i + 1) % N
    j_above = (j + 1) % N
    j_below = (j - 1) % N
    
    # Return neighbors in the specified order
    return [(i_left, j), (i, j_above), (i_right, j), (i, j_below)]


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e