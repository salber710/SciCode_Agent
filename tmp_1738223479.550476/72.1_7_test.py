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
    # Define a helper function that wraps the index using a while loop
    def wrap_index(x, max_val):
        while x < 0:
            x += max_val
        while x >= max_val:
            x -= max_val
        return x
    
    # Calculate neighbors using the helper function
    neighbors = [
        (wrap_index(i - 1, N), j),  # left
        (i, wrap_index(j + 1, N)),  # above
        (wrap_index(i + 1, N), j),  # right
        (i, wrap_index(j - 1, N))   # below
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