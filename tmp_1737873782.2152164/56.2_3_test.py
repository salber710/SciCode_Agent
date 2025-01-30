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




def G_mat(g, pref, dep_order):
    '''Convert to growth rates based on temporal niches
    Input
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Output
    G: "converted" growth rates based on temporal niches, 2d numpy array with dimensions [N, R]
    '''
    N, R = g.shape  # Number of species and resources
    G = np.zeros((N, R))  # Initialize the G matrix

    # Create a mapping from resource index to the order of depletion
    dep_order_index = {resource: i for i, resource in enumerate(dep_order)}

    for i in range(N):  # Iterate over each species
        for j in range(R):  # Iterate over each temporal niche (depletion order)
            # Find the resource in the dep_order corresponding to the j-th temporal niche
            resource_in_niche = dep_order[j]
            
            # Find the column index in the original growth matrix `g` for this resource
            # Resource indices in `pref` are 1-based, so convert to 0-based for indexing
            resource_index = np.where(pref[i] == resource_in_niche)[0][0]
            
            # Assign the growth rate from `g` to the appropriate position in `G`
            G[i, j] = g[i, resource_index]
    
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