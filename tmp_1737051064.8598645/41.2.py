import numpy as np
from math import exp

# Background: In a serially diluted system, resources are consumed by species in a specific order, forming temporal niches. 
# Each species has a preference for certain resources, and they grow exponentially on these resources. The growth rate of 
# each species on each resource is given, and the preference list indicates the order in which species prefer resources. 
# The temporal niches indicate the time available for each resource before it is depleted. The depletion order specifies 
# the sequence in which resources are consumed. The task is to calculate a conversion matrix M, where M[i, j] represents 
# the conversion of biomass from the i-th resource to the j-th species, based on the growth rates and preferences.



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
    
    # Number of species (N) and resources (R)
    N, R = g.shape
    
    # Initialize the conversion matrix M with zeros
    M = np.zeros((R, N))
    
    # Iterate over each species
    for j in range(N):
        # Get the preference list for species j
        species_pref = pref[j]
        
        # Iterate over each resource in the depletion order
        for i, resource_index in enumerate(dep_order):
            # Convert 1-based index to 0-based index
            resource_index -= 1
            
            # Find the position of the current resource in the species' preference list
            preference_position = np.where(species_pref == resource_index + 1)[0][0]
            
            # Calculate the growth rate of species j on the current resource
            growth_rate = g[j, resource_index]
            
            # Calculate the time available for this resource (temporal niche)
            time_available = t[i]
            
            # Calculate the biomass conversion for this resource and species
            # Using exponential growth: biomass = exp(growth_rate * time_available)
            biomass_conversion = exp(growth_rate * time_available)
            
            # Store the conversion in the matrix M
            M[resource_index, j] = biomass_conversion
    
    return M



# Background: In ecological modeling, the concept of structural stability refers to the ability of a community to maintain 
# its structure (i.e., species coexistence) under varying environmental conditions. In the context of resource supply, 
# this is often represented as a simplex where the sum of resource fractions equals one. The conversion matrix M, which 
# represents the conversion of resources to species biomass, can be used to determine the extreme points of this simplex. 
# These extreme points define the boundaries of the region in the resource supply space that supports stable coexistence 
# of the community. The task is to find these extreme points, which are essentially the vertices of the feasibility convex 
# hull in the resource supply space.



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
    
    # Iterate over each species to find the extreme points
    for j in range(N):
        # For each species, find the resource that maximizes the conversion
        # This is done by normalizing the j-th column of M to sum to 1
        # The extreme point is the normalized vector
        max_conversion_index = np.argmax(M[:, j])
        
        # Create a point in the resource supply space
        point = np.zeros(R)
        point[max_conversion_index] = 1.0
        
        # Store this point as one of the extreme points
        res_pts[:, j] = point
    
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
