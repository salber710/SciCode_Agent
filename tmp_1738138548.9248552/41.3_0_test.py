from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from math import exp


def Conversion(g, pref, t, dep_order):
    N, R = g.shape
    M = np.zeros((R, N))
    
    # Map each resource to its depletion time using a dictionary comprehension
    resource_to_time = {dep_order[i] - 1: t[i] for i in range(R)}
    
    # Iterate over each species
    for species_index in range(N):
        # Iterate over each resource based on species preference
        for preference_rank in range(R):
            resource_index = pref[species_index, preference_rank] - 1
            if resource_index in resource_to_time:
                # Calculate the biomass conversion
                growth_rate = g[species_index, resource_index]
                time_available = resource_to_time[resource_index]
                # Calculate the conversion factor using a hyperbolic cosine function for a different approach
                M[resource_index, species_index] = np.cosh(growth_rate * time_available) - 1
    
    return M



def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''
    # Normalize each column of M to sum to 1
    M_normalized = M / np.sum(M, axis=0)
    
    # Calculate the median of the normalized columns
    median = np.median(M_normalized, axis=1)
    
    # Scale the median to lie on the simplex
    median /= np.sum(median)
    
    # Generate points by scaling each axis to reach the simplex boundary
    res_pts = np.eye(M.shape[0]) * median[:, None]
    
    return res_pts



# Background: In ecological systems, the structural stability of a community refers to its ability to maintain coexistence under varying conditions. 
# For a community with N species and R resources, where N=R, the structural stability can be determined by the volume of the feasibility region 
# in the resource supply space. This region is a simplex defined by the condition that the sum of resource fractions equals 1. 
# The area of this region can be calculated using the determinant of the conversion matrix M, which represents the conversion of resources to biomass. 
# The determinant gives the volume of the parallelepiped spanned by the columns of M, and when normalized by the volume of the entire simplex, 
# it provides the fraction of the resource space that supports stable coexistence, i.e., the structural stability.



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
    N, R = g.shape
    M = np.zeros((R, N))
    
    # Map each resource to its depletion time using a dictionary comprehension
    resource_to_time = {dep_order[i] - 1: t[i] for i in range(R)}
    
    # Iterate over each species
    for species_index in range(N):
        # Iterate over each resource based on species preference
        for preference_rank in range(R):
            resource_index = pref[species_index, preference_rank] - 1
            if resource_index in resource_to_time:
                # Calculate the biomass conversion
                growth_rate = g[species_index, resource_index]
                time_available = resource_to_time[resource_index]
                # Calculate the conversion factor
                M[resource_index, species_index] = exp(growth_rate * time_available) - 1
    
    # Calculate the determinant of the conversion matrix M
    det_M = np.linalg.det(M)
    
    # The volume of the entire simplex in R-dimensional space is 1/(R!)
    # For N=R, the structural stability is the determinant of M divided by the volume of the simplex
    simplex_volume = 1 / np.math.factorial(R)
    
    # Structural stability is the fraction of the volume of the feasibility region to the simplex volume
    S = det_M * simplex_volume
    
    return S


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e