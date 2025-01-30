import numpy as np
from math import exp



# Background: In a serially diluted system, resources are consumed by species in a specific order, forming temporal niches. 
# Each species has a preference for certain resources, and they grow exponentially on these resources. The growth rate of a 
# species on a resource is given, and the preference list indicates which resources are preferred by each species. The 
# temporal niches indicate the time duration each resource is available before depletion. The depletion order specifies 
# the sequence in which resources are consumed. The task is to calculate a conversion matrix M, where each element M[i, j] 
# represents the conversion of biomass from the i-th resource to the j-th species, based on the species' initial abundance 
# and growth rate on that resource during its temporal niche.



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
    
    R = g.shape[1]  # Number of resources
    N = g.shape[0]  # Number of species
    
    # Initialize the conversion matrix M with zeros
    M = np.zeros((R, N))
    
    # Iterate over each resource in the depletion order
    for i, resource_index in enumerate(dep_order):
        # Convert 1-based index to 0-based index
        resource_index -= 1
        
        # Get the duration of the temporal niche for this resource
        duration = t[resource_index]
        
        # Calculate the conversion for each species
        for species_index in range(N):
            # Get the growth rate of the species on this resource
            growth_rate = g[species_index, resource_index]
            
            # Calculate the biomass conversion using exponential growth
            # Biomass conversion is exp(growth_rate * duration)
            M[resource_index, species_index] = exp(growth_rate * duration)
    
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
