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

    N, R = G.shape  # Number of species and number of temporal niches
    t = np.zeros(R)  # Array to store the lengths of the temporal niches

    # Solve for the lengths of the temporal niches t_i using the condition that each species' growth leads to a steady state
    for i in range(N):
        # Calculate the sum of growth rates across temporal niches for species i
        sum_G_i = np.sum(G[i])
        
        # Condition for steady state: Product of exp(growth * time) over all niches = Dilution factor D
        # This translates to: exp(sum_G_i * t_total) = D
        # Or: sum_G_i * t_total = log(D)
        t_total = log(D) / sum_G_i

        # Distribute t_total proportionally across temporal niches based on growth rates
        for j in range(R):
            if sum_G_i > 0:
                t[j] += (G[i, j] / sum_G_i) * t_total

    # Check if the calculated temporal niches lengths lead to a feasible steady state
    # This means that each species should be able to grow by the dilution factor D in one complete cycle
    feasible = True
    for i in range(N):
        # Calculate total growth for species i over one cycle
        total_growth = sum(G[i, j] * t[j] for j in range(R))
        if not np.isclose(exp(total_growth), D, atol=1e-6):
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