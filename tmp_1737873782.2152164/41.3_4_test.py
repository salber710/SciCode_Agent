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

    # Calculate the convex hull of the columns of the conversion matrix M
    hull = ConvexHull(M.T)

    # Extract the vertices of the convex hull
    vertices = hull.vertices

    # Select the columns of M that are the vertices of the convex hull
    res_pts = M[:, vertices]

    return res_pts





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

    # Calculate the conversion matrix M using the given Conversion function logic
    N, R = g.shape
    M = np.zeros((R, N))
    dep_order = np.array(dep_order) - 1

    for j in range(N):
        preference = pref[j] - 1

        for i in range(R):
            resource_idx = dep_order[i]

            try:
                preference_rank = np.where(preference == resource_idx)[0][0]
            except IndexError:
                continue

            growth_rate = g[j, resource_idx]
            niche_time = t[i]
            M[resource_idx, j] = exp(growth_rate * niche_time)

    # Calculate the determinant of the conversion matrix M
    det_M = np.linalg.det(M)

    # Calculate the volume of the simplex, which is 1/(R!) for a simplex in R-dimensional space
    volume_of_simplex = 1 / np.math.factorial(R)

    # Structural stability is the fraction of the determinant of M over the volume of the simplex
    S = abs(det_M) / volume_of_simplex

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