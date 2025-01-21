import numpy as np
from math import exp



# Background: In this problem, we are dealing with a serial dilution system where resources are consumed by species in a specific order. Each species has a preference for different resources, and its growth rate varies depending on the resource it consumes. The goal is to compute a conversion matrix M, where each element M[i, j] represents how much biomass from the i-th resource is converted to the j-th species. The growth rates are given in a matrix g, and the preference of each species is given by a matrix pref. The vector t represents the duration of temporal niches where resources are available, and dep_order indicates the order in which resources are depleted. Since species grow exponentially and the consumption yield is 1, the conversion is directly related to the growth rate and the availability time of each resource.



def Conversion(g, pref, t, dep_order):
    '''This function calculates the biomass conversion matrix M
    Inputs:
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    t: temporal niches, 1d numpy array with length R and float elements
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Outputs:
    M: conversion matrix of biomass from resource to species. 2d float numpy array with dimensions [R, N].
    '''

    # Number of species and resources
    N, R = g.shape
    
    # Initialize the conversion matrix M with zeros
    M = np.zeros((R, N), dtype=float)
    
    # Create a map from resource index to its position in the depletion order
    dep_map = {dep_order[i]: i for i in range(R)}
    
    # Iterate over each species
    for j in range(N):
        # Get the preference order for species j
        species_pref = pref[j]
        
        # Iterate over each preferred resource index
        for k in range(R):
            # Resource index from preference list (1-based to 0-based index)
            resource_idx = species_pref[k] - 1
            
            # Find the depletion order position for this resource
            depletion_position = dep_map[species_pref[k]]
            
            # Only consider resources that are not yet depleted
            if depletion_position < R:
                # Calculate the conversion for this resource-species pair
                M[resource_idx, j] = g[j, resource_idx] * t[resource_idx]
                
                # Break after the first non-depleted resource is considered
                break
    
    return M

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('41.1', 3)
target = targets[0]

g = np.array([[1, 0.5], [0.4, 1.1]])
pref = np.array([[1, 2], [2, 1]])
t = np.array([3.94728873, 0.65788146])
dep_order = (2, 1)
assert np.allclose(Conversion(g, pref, t, dep_order), target)
target = targets[1]

g = np.array([[0.82947253, 1.09023245, 1.34105775],
       [0.97056575, 1.01574553, 1.18703424],
       [1.0329076 , 0.82245982, 0.99871483]])
pref = np.array([[3, 2, 1],
       [3, 2, 1],
       [1, 3, 2]])
t = np.array([0.94499274, 1.62433486, 1.88912558])
dep_order = (3, 2, 1)
assert np.allclose(Conversion(g, pref, t, dep_order), target)
target = targets[2]

g = np.array([[1.13829234, 1.10194936, 1.01974872],
       [1.21978402, 0.94386618, 0.90739315],
       [0.97986264, 0.88353569, 1.28083193]])
pref = np.array([[1, 2, 3],
       [1, 2, 3],
       [3, 1, 2]])
t = np.array([2.43030321, 0.26854597, 1.39125344])
dep_order = (3, 1, 2)
assert np.allclose(Conversion(g, pref, t, dep_order), target)
