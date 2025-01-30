from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import itertools
import numpy as np
from math import *



def allowed_orders(pref):
    N, R = pref.shape
    resource_ids = np.arange(1, R + 1)
    all_permutations = permutations(resource_ids)
    allowed_orders_list = []

    # Create a precedence matrix where matrix[i][j] is True if resource i must come before resource j
    precedence_matrix = np.zeros((R, R), dtype=bool)
    for species_pref in pref:
        for i in range(R):
            for j in range(i + 1, R):
                precedence_matrix[species_pref[i] - 1, species_pref[j] - 1] = True

    # Check each permutation against the precedence matrix
    for order in all_permutations:
        if all(precedence_matrix[order[i] - 1, order[j] - 1] for i in range(R) for j in range(i + 1, R)):
            allowed_orders_list.append(order)

    return allowed_orders_list



# Background: In ecological modeling, species growth rates can be influenced by the availability of resources over time. 
# Temporal niches refer to the periods during which specific resources are available for species to exploit. 
# The growth rate of a species in a temporal niche is determined by the availability of its preferred resources in that niche. 
# Given a depletion order of resources, we can map the growth rates based on resources to growth rates based on temporal niches. 
# This involves rearranging the growth rates according to the order in which resources are depleted, reflecting the temporal availability of resources.


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
    dep_order_index = {resource: index for index, resource in enumerate(dep_order)}

    for i in range(N):  # For each species
        for j in range(R):  # For each temporal niche
            # Find the resource that is available in the j-th temporal niche
            resource_in_niche = dep_order[j]
            # Find the index of this resource in the species' preference list
            resource_pref_index = np.where(pref[i] == resource_in_niche)[0][0]
            # Assign the growth rate based on this resource to the temporal niche
            G[i, j] = g[i, resource_pref_index]

    return G


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e