import itertools
import numpy as np
from math import *



# Background: In combinatorial optimization, the task of finding valid permutations of a set of items
# based on certain constraints is common. Here, we are dealing with a problem where we need to find
# all possible depletion orders of resources given preference lists. Each species has a preference
# list indicating the order in which they prefer resources to be depleted. The goal is to generate
# all permutations of resource depletion orders and filter out those that are not possible based on
# the given preference lists. A permutation is considered valid if it respects the order constraints
# implied by the preference lists of all species.




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
                    # If species prefers resource i over j, but j comes before i in the order, it's invalid
                    if species_pref[i] < species_pref[j] and order.index(species_pref[i]) > order.index(species_pref[j]):
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            allowed_orders_list.append(order)

    return allowed_orders_list

from scicode.parse.parse import process_hdf5_to_tuple
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
