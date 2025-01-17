import itertools
import numpy as np
from math import *

# Background: In combinatorial optimization, the task of finding valid permutations of a set of items
# under certain constraints is common. Here, we are dealing with a problem where we need to find all
# possible depletion orders of resources based on given preference lists. Each species has a preference
# list indicating the order in which they prefer resources to be depleted. The goal is to generate all
# possible permutations of resource depletion orders and filter out those that are not allowed based on
# the preference lists. A permutation is allowed if it respects the order constraints implied by the
# preference lists. For example, if a species prefers resource 1 over resource 2, then in any allowed
# permutation, resource 1 must appear before resource 2.




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

    for perm in all_permutations:
        is_allowed = True
        for species_pref in pref:
            for i in range(R):
                for j in range(i + 1, R):
                    if species_pref[i] < species_pref[j]:
                        if perm.index(species_pref[i]) > perm.index(species_pref[j]):
                            is_allowed = False
                            break
                if not is_allowed:
                    break
            if not is_allowed:
                break
        if is_allowed:
            allowed_orders_list.append(perm)

    return allowed_orders_list



# Background: In ecological modeling, species growth rates can be influenced by the availability and order of resource depletion. 
# Temporal niches refer to the time periods during which specific resources are available to species. 
# The growth rate of a species in a temporal niche is determined by the resource available at that time and the species' preference for that resource. 
# The task is to convert growth rates based on resources to growth rates based on temporal niches, using the given resource depletion order. 
# This involves mapping each species' growth rate on a resource to the corresponding temporal niche when that resource is available.


def G_mat(g, pref, dep_order):
    '''Convert to growth rates based on temporal niches
    Input
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Output
    G: "converted" growth rates based on temporal niches, 2d numpy array with dimensions [N, R]
    '''
    N, R = g.shape
    G = np.zeros((N, R))

    # Create a mapping from resource index to its position in the depletion order
    resource_to_niche = {resource: niche for niche, resource in enumerate(dep_order)}

    for i in range(N):  # For each species
        for j in range(R):  # For each resource
            # Find the temporal niche for the j-th preferred resource of species i
            resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
            niche_index = resource_to_niche[resource_index + 1]  # Get the niche index from the depletion order
            # Assign the growth rate of species i on this resource to the corresponding niche
            G[i, niche_index] = g[i, resource_index]

    return G


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('56.2', 3)
target = targets[0]

g = np.array([[1.0, 0.9], [0.8, 1.1]])
pref = np.array([[1, 2], [2, 1]])
dep_order = (1, 2)
assert np.allclose(G_mat(g, pref, dep_order), target)
target = targets[1]

g = np.array([[1.0, 0.9], [0.8, 1.1]])
pref = np.array([[1, 2], [2, 1]])
dep_order = (2, 1)
assert np.allclose(G_mat(g, pref, dep_order), target)
target = targets[2]

g = np.array([[1.0, 0.9, 0.7], [0.8, 1.1, 1.2], [0.3, 1.5, 0.6]])
pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
dep_order = (2, 1, 3)
assert np.allclose(G_mat(g, pref, dep_order), target)
