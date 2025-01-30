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

    # Number of resources and species
    N, R = g.shape

    # Initialize the conversion matrix M with zeros
    M = np.zeros((R, N))
    
    # Create a dictionary to map the depletion order to indices
    dep_indices = {res: idx for idx, res in enumerate(dep_order)}

    # Iterate through each species
    for species in range(N):
        # Iterate through the preference list and match them to depletion order
        for pref_rank in range(R):
            preferred_resource = pref[species, pref_rank]

            # Find where this resource falls in the depletion order
            depletion_index = dep_indices[preferred_resource]

            # Get the growth rate for this species on the preferred resource
            growth_rate = g[species, preferred_resource - 1]

            # Available time in the temporal niches
            available_time = t[depletion_index]

            # Calculate the biomass conversion using exponential growth
            biomass_conversion = exp(growth_rate * available_time) - 1

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