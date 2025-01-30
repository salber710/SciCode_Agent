from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import itertools
import numpy as np
from math import *





def allowed_orders(pref):
    '''Check allowed depletion orders for a set of species with given preference orders
    Input:
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    Output:
    allowed_orders_list: n_allowed by R, list of tuples with int elements between 1 and R. 
    '''
    R = pref.shape[1]  # Number of resources
    all_orders = itertools.permutations(range(1, R + 1))  # Generate all possible depletion orders

    allowed_orders_list = []

    for order in all_orders:
        is_allowed = True
        for species_pref in pref:
            # Check if the first resource in the order is not the last preference for any species
            if order[0] == species_pref[-1]:
                is_allowed = False
                break
        if is_allowed:
            allowed_orders_list.append(order)

    return allowed_orders_list


try:
    targets = process_hdf5_to_tuple('56.1', 3)
    target = targets[0]
    pref = np.array([[1, 2, 3], [2, 1, 3], [3, 1, 2]])
    assert np.allclose(allowed_orders(pref), target)

    target = targets[1]
    pref = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert np.allclose(allowed_orders(pref), target)

    target = targets[2]
    pref = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3]])
    assert np.allclose(allowed_orders(pref), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e