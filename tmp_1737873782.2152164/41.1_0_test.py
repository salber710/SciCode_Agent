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


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e