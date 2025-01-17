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


# Background: In ecological modeling, determining the feasibility of a steady state of coexistence involves checking if the growth rates of species in different temporal niches can balance out the dilution factor. The dilution factor represents the rate at which resources are removed or diluted in the system. For a steady state to be feasible, each species must be able to maintain its population in each temporal niche, meaning that the growth rate in each niche must be at least equal to the dilution factor. The lengths of the temporal niches, denoted as t_i, are determined by the balance between growth rates and the dilution factor. If all species can coexist with positive niche lengths, the system is considered feasible.

def check_G_feasibility(G, D):
    '''Determine if a "converted" growth rate matrix G leads to a feasible coexistence. 
    Input 
    G: growth rate based on temporal niches, 2d numpy float array with dimensions [N, R]
    D: dilution factor, float
    Output
    feasible: boolean
    '''
    N, R = G.shape

    # Initialize a list to store the lengths of temporal niches
    t = np.zeros(R)

    # Check if there exists a set of positive niche lengths t_i such that G[i, j] >= D for all i, j
    for j in range(R):
        # For each temporal niche, check if there is at least one species with a growth rate >= D
        if np.any(G[:, j] >= D):
            # If there is, set the length of this niche to a positive value
            t[j] = 1  # Arbitrary positive value, since we only care about feasibility
        else:
            # If no species can maintain its population in this niche, the system is not feasible
            return False

    # If all niches have positive lengths, the system is feasible
    return True



# Background: In ecological modeling, determining feasible depletion orders involves finding permutations of resource depletion that allow for a steady state of coexistence among species. This requires that for each depletion order, the growth rates of species in the resulting temporal niches can balance out the dilution factor. A depletion order is feasible if, for each temporal niche, there is at least one species whose growth rate in that niche is greater than or equal to the dilution factor. This ensures that all species can maintain their populations across the temporal niches.




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
    all_permutations = itertools.permutations(range(1, R + 1))
    possible_orders = []

    for dep_order in all_permutations:
        # Convert growth rates based on resources to growth rates based on temporal niches
        G = np.zeros((N, R))
        resource_to_niche = {resource: niche for niche, resource in enumerate(dep_order)}

        for i in range(N):  # For each species
            for j in range(R):  # For each resource
                resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
                niche_index = resource_to_niche[resource_index + 1]  # Get the niche index from the depletion order
                G[i, niche_index] = g[i, resource_index]

        # Check if this depletion order leads to a feasible coexistence
        feasible = True
        for j in range(R):
            if not np.any(G[:, j] >= D):
                feasible = False
                break

        if feasible:
            possible_orders.append(dep_order)

    return possible_orders


from scicode.parse.parse import process_hdf5_to_tuple

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
