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



# Background: In ecological modeling, the concept of structural stability refers to the ability of a community to maintain its structure (i.e., species coexistence) under varying environmental conditions. The resource supply space is a simplex where the sum of resource fractions equals one. The feasibility region within this space is determined by the conversion matrix M, which represents how resources are converted into biomass for different species. The extreme points of this region, or the vertices of the convex hull, indicate the combinations of resource supplies that allow for stable coexistence of all species. These points can be found using linear algebra techniques, specifically by identifying the vertices of the convex hull formed by the columns of the matrix M.



def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''
    # Transpose M to get the points in the correct format for ConvexHull
    points = M.T
    
    # Compute the convex hull of the points
    hull = ConvexHull(points)
    
    # Extract the vertices of the convex hull
    vertices = hull.vertices
    
    # Select the points corresponding to the vertices
    res_pts = points[vertices]
    
    # Transpose back to the original format
    res_pts = res_pts.T
    
    return res_pts


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e