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



# Background: In ecological modeling, the structural stability of a community refers to its ability to maintain species 
# coexistence under varying environmental conditions. This is often represented as a region within a resource supply 
# simplex, where the sum of resource fractions equals one. For a system where the number of species (N) equals the number 
# of resources (R), the area of this region can be determined by calculating the determinant of the conversion matrix M. 
# The determinant provides a measure of the volume (or area in 2D) of the parallelepiped formed by the columns of M. 
# The fraction of this area within the whole simplex represents the structural stability of the community.



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
