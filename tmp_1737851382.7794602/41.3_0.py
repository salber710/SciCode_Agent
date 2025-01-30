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
        max_resource_indices = np.argwhere(M[:, j] == np.max(M[:, j])).flatten()
        
        # Set the corresponding resource supply points
        if len(max_resource_indices) == R:
            # If all resources are equally good, distribute evenly across all
            res_pts[:, j] = 1 / R
        else:
            # Otherwise, distribute evenly among the best resources
            res_pts[max_resource_indices, j] = 1.0 / len(max_resource_indices)
    
    return res_pts



# Background: In ecological modeling, the structural stability of a community refers to its ability to maintain species 
# coexistence under varying environmental conditions. The resource supply space is represented as a simplex where the sum 
# of resource fractions equals one. When the number of resources (R) equals the number of species (N), the area of the 
# region formed by the extreme points in this space can be calculated using the determinant of the conversion matrix M. 
# This determinant gives the volume of the parallelepiped spanned by the columns of M, which corresponds to the area of 
# the region in the simplex. The fraction of this area within the whole resource supply simplex represents the structural 
# stability of the community.



def StrucStability(g, pref, t, dep_order):
    '''This function gets the community's structural stability
    Inputs:
    g: growth rates based on resources, 2d numpy array with dimensions [N, R] and float elements
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    t: temporal niches, 1d numpy array with length R and float elements
    dep_order: resource depletion order, a tuple of length R with int elements between 1 and R
    Outputs:
    S: structural stability of the community, float
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
    
    # Calculate the determinant of the conversion matrix M
    det_M = np.linalg.det(M)
    
    # The structural stability S is the absolute value of the determinant
    S = abs(det_M)
    
    return S

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('41.3', 3)
target = targets[0]

g = np.array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])
pref = np.array([[1, 2, 3],
       [2, 1, 3],
       [3, 1, 2]])
dep_order = (1, 2, 3)
t = np.array([1, 0, 0])
assert np.allclose(StrucStability(g, pref, t, dep_order), target)
target = targets[1]

g = np.array([[0.68879706, 0.8834816 , 0.70943619],
       [1.04310011, 0.8411964 , 0.86002165],
       [0.97550015, 0.84997877, 1.04842294]])
pref = np.array([[2, 3, 1],
       [1, 3, 2],
       [3, 1, 2]])
dep_order = (3, 1, 2)
t = np.array([0.51569821, 0.57597405, 4.12085303])
assert np.allclose(StrucStability(g, pref, t, dep_order), target)
target = targets[2]

g = np.array([[0.79099249, 1.00928232, 0.90901695, 1.07388973],
       [0.89646902, 0.79124502, 0.79294553, 1.18807732],
       [0.78464268, 1.04435014, 0.97980406, 1.00469375],
       [0.85474971, 0.9244668 , 1.27430835, 0.47863501]])
pref = np.array([[4, 2, 3, 1],
       [4, 1, 3, 2],
       [2, 4, 3, 1],
       [3, 2, 1, 4]])
dep_order = (4, 3, 1, 2)
t = np.array([1.51107846, 0.88238109, 1.58035451, 0.43578957])
assert np.allclose(StrucStability(g, pref, t, dep_order), target)
