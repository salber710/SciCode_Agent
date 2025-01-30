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

    N, R = G.shape  # Number of species (N) and number of temporal niches/resources (R)
    
    # Initialize array to store lengths of temporal niches
    t = np.zeros(R)
    
    # Calculate the sum of growth rates for each species to determine if coexistence is possible
    for i in range(N):  # For each species
        growth_sum = sum(G[i, j] for j in range(R))
        
        # Calculate the time t_i for each temporal niche
        for j in range(R):
            if G[i, j] > 0:
                # Calculate the proportion of time spent in this niche based on growth rates
                t[j] += G[i, j] / growth_sum
                
    # Normalize the time array to ensure it sums to 1 (complete cycle)
    t /= np.sum(t)
    
    # Check feasibility: each species must grow by exactly the dilution factor D over one cycle
    feasible = True
    for i in range(N):
        growth_factor = 1.0
        for j in range(R):
            growth_factor *= exp(G[i, j] * t[j])
        
        # Check if the growth factor for this species over the cycle is equal to D
        if not np.isclose(growth_factor, D, atol=1e-6):
            feasible = False
            break
    
    return feasible


try:
    targets = process_hdf5_to_tuple('56.3', 3)
    target = targets[0]
    G = np.array([[1. , 0.9],
           [1.1, 1.1]])
    D = 100.0
    assert (check_G_feasibility(G, D)) == target

    target = targets[1]
    G = np.array([[1. , 1. ],
           [1.1, 0.8]])
    D = 20.0
    assert (check_G_feasibility(G, D)) == target

    target = targets[2]
    G = np.array([[1. , 1. , 0.7],
           [1.1, 1.2, 1.2],
           [0.6, 0.6, 0.6]])
    D = 100.0
    assert (check_G_feasibility(G, D)) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e