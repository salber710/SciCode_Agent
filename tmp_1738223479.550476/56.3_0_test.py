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



# Background: In ecological modeling, the concept of temporal niches is used to describe how species can coexist by utilizing resources at different times. The "converted" growth rate matrix G describes how well each species grows in each temporal niche. The dilution factor D represents the rate at which population sizes are reduced, possibly due to environmental factors. To determine whether a system reaches a feasible steady state of coexistence, we need to calculate the lengths of time each temporal niche is viable, denoted as t_i, for the system's growth rates to balance the dilution factor. The feasibility is determined by checking if all species have non-negative growth in all niches when accounting for D.

def check_G_feasibility(G, D):
    '''Determine if a "converted" growth rate matrix G leads to a feasible coexistence. 
    Input 
    G: growth rate based on temporal niches, 2d numpy float array with dimensions [N, R]
    D: dilution factor, float
    Output
    feasible: boolean
    '''
    N, R = G.shape
    
    # Initialize the lengths of temporal niches t_i
    t = np.zeros(R)
    
    # Attempt to solve the system of equations for the niche lengths
    try:
        # Construct the matrix and vector for the linear system G * t = D
        # We have a system G * t = D * e, where e is a vector of ones
        A = G - D * np.eye(N)
        
        # Solve the system A * t = 0 for t using least squares method
        # Since we are solving for a homogeneous system, we use the null space
        U, s, Vt = np.linalg.svd(A)
        t = Vt[-1, :]  # Take the last row of V^T (corresponds to the smallest singular value)
        
        # Check if all elements of t are non-negative to determine feasibility
        feasible = np.all(t >= 0)
        
    except np.linalg.LinAlgError:
        # If the system cannot be solved, it's not feasible
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