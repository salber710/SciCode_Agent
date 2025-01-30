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

    N, R = g.shape  # Number of species (N) and number of resources (R)
    G = np.zeros((N, R))  # Initialize the output array with zeros

    # Create a dictionary to map resource index to its depletion order index
    dep_order_index = {resource: index for index, resource in enumerate(dep_order)}

    for i in range(N):  # For each species
        # Get the preference order for the current species
        species_pref = pref[i]
        # Create a mapping from the preference order to the depletion order
        pref_to_dep_order = [dep_order_index[resource] for resource in species_pref]

        # Sort the growth rates according to the depletion order determined by the preference
        for j in range(R):
            # The j-th temporal niche corresponds to the j-th place in the depletion order
            # Find the resource index in the preference list that corresponds to this niche
            resource_index = pref_to_dep_order.index(j)
            # Assign the growth rate for this niche
            G[i, j] = g[i, species_pref[resource_index] - 1]

    return G


def check_G_feasibility(G, D):
    '''Determine if a "converted" growth rate matrix G leads to a feasible coexistence. 
    Input 
    G: growth rate based on temporal niches, 2d numpy float array with dimensions [N, R]
    D: dilution factor, float
    Output
    feasible: boolean
    '''
    N, R = G.shape

    # Initialize a list to hold the lengths of each temporal niche
    t = np.zeros(R)

    # Calculate the lengths of each temporal niche
    for j in range(R):
        # Calculate total growth in temporal niche j by summing the growth rates
        total_growth = np.sum(G[:, j])
        
        if total_growth == 0:
            # If total growth is zero in a temporal niche, it's not feasible
            return False

        # Calculate the time length for niche j using the formula t_j = ln(D) / total_growth
        t[j] = log(D) / total_growth

    # Check if any temporal niche has a negative time length, which would be infeasible
    if np.any(t < 0):
        return False

    # If all time lengths are non-negative, the coexistence is feasible
    return True



def get_dep_orders(g, pref, D):
    '''filter for feasible depletion orders
    Input 
    g:         growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref:      species' preference order, 2d numpy array with dimensions [N, R] and int elements
    D:    dilution factor, float
    Output
    possible_orders: all possible depletion orders, a list of tuples with int elements
    '''
    R = g.shape[1]  # Number of resources
    all_orders = itertools.permutations(range(1, R + 1))  # Generate all possible depletion orders

    possible_orders = []

    for dep_order in all_orders:
        # Convert growth rates based on resources to growth rates based on temporal niches
        G = G_mat(g, pref, dep_order)

        # Check if the current depletion order results in a feasible coexistence
        if check_G_feasibility(G, D):
            possible_orders.append(dep_order)

    return possible_orders


try:
    targets = process_hdf5_to_tuple('56.4', 3)
    target = targets[0]
    g = np.array([[1.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.9]])
    pref = np.argsort(-g, axis=1) + 1
    D = 100
    assert np.allclose(get_dep_orders(g, pref, D), target)

    target = targets[1]
    g = np.array([[1.0, 0.8, 0.9, 0.7], 
                  [0.9, 0.78, 1.01, 0.1],
                  [0.92, 0.69, 1.01, 0.79], 
                  [0.65, 0.94, 0.91, 0.99]])
    pref = np.argsort(-g, axis=1) + 1
    D = 100
    assert np.allclose(get_dep_orders(g, pref, D), target)

    target = targets[2]
    g = np.array([[1.0, 0.8, 0.9, 0.7], 
                  [0.9, 0.78, 1.01, 0.1],
                  [0.92, 0.69, 1.01, 0.79], 
                  [0.65, 0.94, 0.91, 0.99]])
    pref = np.array([[1, 2, 3, 4], 
                     [2, 3, 4, 1], 
                     [3, 4, 1, 2], 
                     [4, 1, 2, 3]])
    D = 100
    assert np.allclose(get_dep_orders(g, pref, D), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e