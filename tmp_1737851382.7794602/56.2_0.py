import itertools
import numpy as np
from math import *

# Background: In combinatorial optimization, the task of finding valid permutations of a set of items
# based on certain constraints is common. Here, we are dealing with a problem where we need to find
# all possible depletion orders of resources given preference lists. Each species has a preference
# list indicating the order in which they prefer resources to be depleted. The goal is to generate
# all permutations of resource depletion orders and filter out those that are not possible based on
# the given preference lists. A permutation is considered valid if it respects the order constraints
# implied by the preference lists of all species.




def allowed_orders(pref):
    '''Check allowed depletion orders for a set of species with given preference orders
    Input:
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements
    Output:
    allowed_orders_list: n_allowed by R, list of tuples with int elements between 1 and R. 
    '''
    if pref.size == 0:
        return []
    
    if not np.issubdtype(pref.dtype, np.integer):
        raise TypeError("Preference matrix must contain integer values.")
    
    try:
        N, R = pref.shape
    except ValueError:
        raise ValueError("Preference matrix must be a 2D array.")
    
    if R == 0:
        return []
    
    # Normalize preferences to start from 1 to R
    unique_elements = np.unique(pref)
    normalized_pref = np.zeros_like(pref)
    for i, element in enumerate(unique_elements):
        normalized_pref[pref == element] = i + 1
    
    all_permutations = itertools.permutations(range(1, R + 1))
    allowed_orders_list = []

    for order in all_permutations:
        valid = True
        for species_pref in normalized_pref:
            # Create a rank list where index is resource-1 and value is the preference rank
            rank = [0] * R
            for i in range(R):
                rank[species_pref[i] - 1] = i
            
            # Check if the order respects the rank (preference)
            for i in range(R):
                for j in range(i + 1, R):
                    if rank[order[i] - 1] > rank[order[j] - 1]:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            allowed_orders_list.append(order)

    return allowed_orders_list



# Background: In ecological modeling, species growth rates can be influenced by the availability of resources over time.
# Temporal niches refer to the periods during which specific resources are available for species to exploit. The growth
# rate of a species in a temporal niche is determined by the availability of its preferred resources in that niche.
# Given a depletion order of resources, we can map the growth rates based on resources to growth rates based on temporal
# niches. This involves rearranging the growth rates according to the order in which resources are depleted, respecting
# each species' preference list.


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
    dep_order_index = {resource: i for i, resource in enumerate(dep_order)}
    
    for i in range(N):  # For each species
        for j in range(R):  # For each temporal niche
            # Find the resource that is available in the j-th temporal niche
            resource_in_niche = dep_order[j]
            
            # Find the index of this resource in the species' preference list
            # Since pref is 1-indexed, we need to adjust by subtracting 1
            resource_pref_index = np.where(pref[i] == resource_in_niche)[0][0]
            
            # Assign the growth rate for this species in this temporal niche
            G[i, j] = g[i, resource_pref_index]
    
    return G

from scicode.parse.parse import process_hdf5_to_tuple
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
