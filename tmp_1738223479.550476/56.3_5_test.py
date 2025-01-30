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

    def is_valid_partial_order(partial_order, pref):
        # Validate the current partial order against all species preferences
        resource_positions = {resource: i for i, resource in enumerate(partial_order)}
        for species_pref in pref:
            for i in range(len(species_pref)):
                for j in range(i + 1, len(species_pref)):
                    if species_pref[i] in resource_positions and species_pref[j] in resource_positions:
                        if resource_positions[species_pref[i]] > resource_positions[species_pref[j]]:
                            return False
        return True

    R = pref.shape[1]
    resources = list(range(1, R + 1))
    allowed_orders_list = []

    def generate_orders(partial_order, remaining_resources):
        if not remaining_resources:
            allowed_orders_list.append(tuple(partial_order))
            return

        for i in range(len(remaining_resources)):
            next_order = partial_order + [remaining_resources[i]]
            if is_valid_partial_order(next_order, pref):
                generate_orders(next_order, remaining_resources[:i] + remaining_resources[i+1:])

    generate_orders([], resources)
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
    
    N, R = g.shape
    G = np.empty((N, R))
    
    # Convert dep_order to zero-indexed
    dep_order_zero_indexed = [resource - 1 for resource in dep_order]
    
    # Build a lookup table for depletion order
    depletion_lookup = np.zeros(R, dtype=int)
    for index, resource in enumerate(dep_order_zero_indexed):
        depletion_lookup[resource] = index
    
    for species_idx in range(N):
        # Translate preferences to zero-indexed
        zero_indexed_pref = pref[species_idx] - 1
        
        # Create an array that will store the depletion order for each preferred resource
        resource_depletion_order = np.empty(R, dtype=int)
        for pref_idx, resource in enumerate(zero_indexed_pref):
            resource_depletion_order[pref_idx] = depletion_lookup[resource]
        
        # Sort preferences by the actual depletion order
        sorted_pref_indices = np.argsort(resource_depletion_order)
        
        # Populate G using the sorted preferences
        for niche_idx, pref_idx in enumerate(sorted_pref_indices):
            resource_idx = zero_indexed_pref[pref_idx]
            G[species_idx, niche_idx] = g[species_idx, resource_idx]
    
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
    
    # Construct the matrix equation (G - D*I) * t = 0, with t >= 0
    A = G - D * np.eye(N)
    
    # Since we want to solve A * t = 0 with t >= 0, we can use a linear least squares solver with non-negativity constraints
    # Set up an optimization problem to minimize the norm of A * t subject to t >= 0
    try:
        # Using lsq_linear to solve the non-negative least squares problem
        result = lsq_linear(A, np.zeros(N), bounds=(0, np.inf))
        
        # Check if the solution is feasible
        feasible = result.success and np.all(result.x >= 0)

    except Exception as e:
        # If there's any issue with the optimization, assume infeasibility
        feasible = False

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