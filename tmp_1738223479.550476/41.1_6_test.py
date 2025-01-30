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

    R = g.shape[1]  # Number of resources
    N = g.shape[0]  # Number of species

    # Initialize the conversion matrix M with zeros
    M = np.zeros((R, N))

    # Create a reverse lookup for resource depletion order
    resource_to_time_index = {resource: i for i, resource in enumerate(dep_order)}

    # Iterate over each species
    for species in range(N):
        # Iterate over each resource according to species preference
        for pref_rank in range(R):
            # Find the preferred resource from the preference list
            preferred_resource = pref[species, pref_rank]

            # Get the resource's depletion time index
            depletion_time_index = resource_to_time_index[preferred_resource]

            # Calculate growth rate using the preferred resource
            growth_rate = g[species, preferred_resource - 1]

            # Fetch the time available in the temporal niche for this resource
            time_available = t[depletion_time_index]

            # Compute the biomass conversion using exponential growth
            biomass_conversion = exp(growth_rate * time_available) - 1

            # Update the conversion matrix
            M[preferred_resource - 1, species] = biomass_conversion

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