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




def check_G_feasibility(G, D):
    N, R = G.shape

    # Objective function: minimize the sum of squared t_i to encourage smaller and balanced niche times
    def objective(t):
        return np.sum(t**2)

    # Constraints: G @ t should be greater than or equal to D for each species
    constraints = [{'type': 'ineq', 'fun': lambda t: np.dot(G, t) - D}]

    # Initial guess: start with equal distribution of niche times
    t0 = np.ones(R) / R

    # Bounds for t_i (all t_i must be non-negative)
    bounds = [(0, None) for _ in range(R)]

    # Minimize the sum of squared t_i
    result = minimize(objective, t0, bounds=bounds, constraints=constraints)

    # Check if the solution found is feasible
    feasible = result.success and np.all(np.dot(G, result.x) >= D)

    return feasible



# Background: In ecological modeling, determining feasible depletion orders is crucial for understanding how species can coexist in a shared environment. Each species has a preference for certain resources, and these resources are depleted over time. The order in which resources are depleted affects the growth rates of species, which must be sufficient to overcome a given dilution factor for coexistence to be feasible. The task is to find all possible depletion orders that allow for a feasible steady state, where the growth rates in each temporal niche, when multiplied by the time spent in that niche, meet or exceed the dilution factor.




def get_dep_orders(g, pref, D):
    '''filter for feasible depletion orders
    Input 
    g:         growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref:      species' preference order, 2d numpy array with dimensions [N, R] and int elements
    D:    dilution factor, float
    Output
    possible_orders: all possible depletion orders, a list of tuples with int elements
    '''
    
    N, R = g.shape
    resource_ids = np.arange(1, R + 1)
    all_permutations = itertools.permutations(resource_ids)
    possible_orders = []

    # Function to convert growth rates based on resources to temporal niches
    def G_mat(g, pref, dep_order):
        G = np.zeros((N, R))
        dep_map = {dep_order[i]: i for i in range(R)}

        for i in range(N):
            for j in range(R):
                resource_idx = pref[i, j] - 1
                niche_idx = dep_map[dep_order[j]]
                G[i, niche_idx] += g[i, resource_idx] / R

        return G

    # Check feasibility of each depletion order
    for dep_order in all_permutations:
        G = G_mat(g, pref, dep_order)

        # Objective function: minimize the sum of squared t_i
        def objective(t):
            return np.sum(t**2)

        # Constraints: G @ t should be greater than or equal to D for each species
        constraints = [{'type': 'ineq', 'fun': lambda t: np.dot(G, t) - D}]

        # Initial guess: start with equal distribution of niche times
        t0 = np.ones(R) / R

        # Bounds for t_i (all t_i must be non-negative)
        bounds = [(0, None) for _ in range(R)]

        # Minimize the sum of squared t_i
        result = minimize(objective, t0, bounds=bounds, constraints=constraints)

        # Check if the solution found is feasible
        if result.success and np.all(np.dot(G, result.x) >= D):
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