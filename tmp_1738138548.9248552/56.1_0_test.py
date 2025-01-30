from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import itertools
import numpy as np
from math import *



# Background: In combinatorial optimization, the task of finding valid permutations of a set of items
# under certain constraints is common. Here, we are dealing with the problem of finding all possible
# depletion orders of resources given preference lists. Each species has a preference list indicating
# the order in which it prefers resources to be depleted. The goal is to generate all permutations of
# resource depletion orders and filter out those that are not possible given the preference constraints.
# A depletion order is valid if it respects the preference order of each species, meaning that if a
# species prefers resource A over resource B, then in the depletion order, A must appear before B.




def allowed_orders(pref):
    '''Check allowed depletion orders for a set of species with given preference orders
    Input:
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    Output:
    allowed_orders_list: n_allowed by R, list of tuples with int elements between 1 and R. 
    '''
    N, R = pref.shape
    all_permutations = itertools.permutations(range(1, R + 1))
    allowed_orders_list = []

    for order in all_permutations:
        valid = True
        for species_pref in pref:
            for i in range(R):
                for j in range(i + 1, R):
                    if species_pref[i] < species_pref[j] and order.index(species_pref[i]) > order.index(species_pref[j]):
                        valid = False
                        break
                if not valid:
                    break
        if valid:
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