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
    # Create a dictionary to map directions to their corresponding index changes
    direction_map = {
        'left': (-1, 0),
        'above': (0, 1),
        'right': (1, 0),
        'below': (0, -1)
    }
    # Use a loop to calculate neighbors using the direction_map
    neighbors = []
    for direction in ['left', 'above', 'right', 'below']:
        di, dj = direction_map[direction]
        neighbor_i = (i + di + N) % N  # Add N to ensure positive mod result
        neighbor_j = (j + dj + N) % N
        neighbors.append((neighbor_i, neighbor_j))
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