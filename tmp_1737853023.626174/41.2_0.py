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
    
    # Check for dimension consistency
    if pref.shape != g.shape:
        raise ValueError("The dimensions of growth rates and preferences must match.")
    
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



# Background: In ecological modeling, the concept of structural stability refers to the ability of a community to maintain 
# its structure (i.e., species coexistence) under varying environmental conditions, such as changes in resource supply. 
# The resource supply space can be represented as a simplex where the sum of resource fractions equals one. The conversion 
# matrix M, which describes how resources are converted into species biomass, can be used to determine the extreme points 
# of this space that support stable coexistence. These extreme points form a convex hull in the resource supply space, 
# representing the boundaries of feasible resource distributions that allow for the coexistence of all species.



def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''
    # Number of resources (R) and species (N)
    R, N = M.shape
    
    # Initialize the array to store the extreme points
    res_pts = np.zeros((R, N))
    
    # For each species, find the resource supply point that maximizes its growth
    for j in range(N):
        # The j-th column of M gives the conversion rates for species j across all resources
        # We want to find the resource that maximizes this conversion rate for species j
        max_resource_index = np.argmax(M[:, j])
        
        # Set the corresponding resource supply point
        res_pts[max_resource_index, j] = 1.0
    
    return res_pts

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('41.2', 3)
target = targets[0]

M = np.array([[99.0000004 , 23.13753917],
       [ 0.        , 75.86246093]])
assert np.allclose(GetResPts(M), target)
target = targets[1]

M = np.array([[79.13251071, 84.01501987, 98.99999879],
       [17.31627415, 12.91479347,  0.        ],
       [ 2.55121514,  2.07018782,  0.        ]])
assert np.allclose(GetResPts(M), target)
target = targets[2]

M = np.array([[20.5867424 , 25.89695551,  6.76786984],
       [78.41325762, 73.10304309, 70.74799474],
       [ 0.        ,  0.        , 21.48413504]])
assert np.allclose(GetResPts(M), target)
