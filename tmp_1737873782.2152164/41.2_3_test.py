from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from math import exp



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

    # Map depletion order from 1-based indexing to 0-based indexing
    dep_order = np.array(dep_order) - 1

    # Iterate over each species
    for j in range(N):
        # Get the preference order for species j and convert it to 0-based indexing
        preference = pref[j] - 1

        # Iterate over the depletion order
        for i in range(R):
            # Get the current resource being depleted
            resource_idx = dep_order[i]

            # Find where this resource ranks in the species j's preference
            try:
                preference_rank = np.where(preference == resource_idx)[0][0]
            except IndexError:
                # If the resource is not in the preference list, skip it
                continue

            # Calculate the growth for species j on resource i
            growth_rate = g[j, resource_idx]
            niche_time = t[i]

            # Convert the biomass by the growth factor over the niche time
            M[resource_idx, j] = exp(growth_rate * niche_time)

    return M





def GetResPts(M):
    '''This function finds the endpoints of the feasibility convex hull
    Inputs:
    M: conversion matrix of biomass, 2d float numpy array of size [R, N]
    Outputs:
    res_pts: a set of points in the resource supply space that marks the region of feasibility. 2d float numpy array of size [R, N].
    '''

    # Number of resources (R) and species (N)
    R, N = M.shape

    # Initialize an array to hold the resource points in the resource supply space
    res_pts = np.zeros((R, N))

    # Iterate over each species to find the extreme points in the resource supply space
    for j in range(N):
        # The j-th column in M represents the conversion factors for the j-th species
        # Normalize this column to represent a point in the simplex of resource supply fractions
        col_sum = np.sum(M[:, j])
        if col_sum > 0:
            res_pts[:, j] = M[:, j] / col_sum
        else:
            # If the sum is zero, we can't normalize; it implies no conversion for this species
            # Set this column to a default value (e.g., evenly distribute among resources)
            res_pts[:, j] = np.ones(R) / R

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