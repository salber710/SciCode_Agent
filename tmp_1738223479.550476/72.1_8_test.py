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
    # Using a dictionary to store neighbor positions and a custom wrapping function
    def periodic_wrap(index, max_value):
        return (index + max_value) % max_value
    
    neighbor_positions = {
        'left': (periodic_wrap(i - 1, N), j),
        'above': (i, periodic_wrap(j + 1, N)),
        'right': (periodic_wrap(i + 1, N), j),
        'below': (i, periodic_wrap(j - 1, N))
    }
    
    # Extracting neighbors in the specified order
    return [neighbor_positions[dir] for dir in ['left', 'above', 'right', 'below']]


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