import numpy as np
from math import exp

# Background: 
# In a serial dilution system with multiple species and resources, each species consumes resources in a specific order of preference.
# The process involves calculating how each resource contributes to the growth of each species during a cycle.
# Growth is exponential and determined by specific growth rates for each resource-species pair.
# The conversion matrix M[i, j] indicates the biomass conversion from the i-th resource to the j-th species.
# Given the growth rates, species' preferences, temporal niches, and depletion order, we can calculate how much of each resource contributes to the growth of each species.
# The growth of a species on a resource is given by exp(growth_rate * time_in_niche), assuming a yield of 1.



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
    
    # Initialize the conversion matrix with zeros
    M = np.zeros((R, N))
    
    # Map depletion order to zero-based index
    dep_index = [d - 1 for d in dep_order]
    
    # Iterate over each resource in depletion order
    for i, resource in enumerate(dep_index):
        # Calculate the time in the current temporal niche
        time_in_niche = t[i]
        
        # For each species, calculate the biomass conversion from this resource
        for species in range(N):
            # Find the actual growth rate for this species on the current resource
            growth_rate = g[species, resource]
            
            # Calculate the biomass conversion
            biomass_conversion = exp(growth_rate * time_in_niche)
            
            # Update the conversion matrix
            M[resource, species] = biomass_conversion
    
    return M



# Background: In the context of ecological modeling, the structural stability of a community can be analyzed by examining the resource supply space, which is represented as a simplex. The simplex is defined by the constraint ∑R_i = 1, where R_i denotes the fraction of resource i. The convex hull of points in this space that support stable coexistence of species is determined by the conversion matrix M. The matrix M represents how efficiently each resource is converted into biomass for each species. The task is to find the extreme points in this supply space, which can be thought of as the vertices of the convex hull, indicating the boundaries of conditions under which all species can coexist.


def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''
    # Linear algebra approach to find the extreme points of the convex hull
    # The extreme points are essentially the vertices of the convex hull in the resource supply space

    # Compute the pseudo-inverse of M
    M_pinv = np.linalg.pinv(M)

    # Identity matrix of size N (number of species)
    I = np.eye(M.shape[1])

    # Calculate the extreme points in the resource supply space
    res_pts = M_pinv @ I

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
