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



def G_mat(g, pref, dep_order):
    N, R = g.shape
    G = np.zeros((N, R))
    dep_map = {dep_order[i]: i for i in range(R)}

    # Calculate the growth rate for each species in each temporal niche
    for i in range(N):
        for j in range(R):
            # Get the resource index from preference list, adjusted for zero-indexing
            resource_idx = pref[i, j] - 1
            # Map the resource to its niche based on depletion order
            niche_idx = dep_map[dep_order[j]]
            # Assign the growth rate to the corresponding niche, taking the average if multiple preferences map to the same niche
            G[i, niche_idx] += g[i, resource_idx] / R

    return G



# Background: In ecological modeling, the concept of temporal niches refers to the partitioning of time into distinct periods where different species can exploit resources. The growth rate matrix G represents the growth rates of species in these temporal niches. A feasible steady state of coexistence occurs when all species can maintain a positive growth rate over time, considering the dilution factor D, which represents the rate at which resources are removed or diluted. To determine feasibility, we need to solve for the lengths of the temporal niches (t_i) such that the average growth rate across all niches is non-negative for each species. This involves solving a linear system where the sum of growth rates weighted by niche lengths equals the dilution factor.


def check_G_feasibility(G, D):
    '''Determine if a "converted" growth rate matrix G leads to a feasible coexistence. 
    Input 
    G: growth rate based on temporal niches, 2d numpy float array with dimensions [N, R]
    D: dilution factor, float
    Output
    feasible: boolean
    '''
    N, R = G.shape
    
    # Create the matrix A and vector b for the linear system A * t = b
    A = G.T  # Transpose of G, so each column corresponds to a species
    b = np.full(N, D)  # Vector b, where each element is the dilution factor D
    
    # Solve the linear system A * t = b for t
    try:
        t = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # If the matrix A is singular, the system cannot be solved
        return False
    
    # Check if all t_i are positive, which is required for a feasible solution
    feasible = np.all(t > 0)
    
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