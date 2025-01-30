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
    # Use lambda functions to calculate neighbors with periodic wrapping
    wrap = lambda x: x % N
    neighbors = [
        (wrap(i - 1), j),  # left
        (i, wrap(j + 1)),  # above
        (wrap(i + 1), j),  # right
        (i, wrap(j - 1))   # below
    ]
    return neighbors


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