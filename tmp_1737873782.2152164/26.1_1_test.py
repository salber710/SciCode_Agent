from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy



def SpeciesGrowth(g, pref, Rs, alive):
    '''This function calculates the species growth rate
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    Rs: resource level in environment. 1d float numpy array of length R. 
    alive: whether the species is present or not. 1d boolean numpy array of length N. 
    Outputs: 
    g_temp: current growth rate of species, 1D float numpy array of length N. 
    r_temp: list of resources that each species is eating. 1D int numpy array of length N. 
    '''
    
    N, R = g.shape
    g_temp = np.zeros(N)
    r_temp = np.zeros(N, dtype=int)
    
    # Iterate over each species
    for i in range(N):
        if not alive[i]:
            # If the species is not present, it has zero growth rate and consumes no resources
            g_temp[i] = 0
            r_temp[i] = 0
            continue
        
        # Check preferences and available resources
        for j in range(R):
            resource_index = pref[i, j] - 1  # Convert preference index to 0-based resource index
            if Rs[resource_index] > 0:
                # Resource is available, calculate growth rate
                g_temp[i] = g[i, resource_index]
                r_temp[i] = resource_index + 1  # Store the 1-based index of the resource being consumed
                break
        else:
            # If no preferred resources are available, the species consumes nothing
            g_temp[i] = 0
            r_temp[i] = 0
    
    return g_temp, r_temp


try:
    targets = process_hdf5_to_tuple('26.1', 3)
    target = targets[0]
    g = np.array([[1.0, 0.9], [0.8, 1.1]])
    pref = np.array([[1, 2], [2, 1]])
    Rs = np.array([0, 0])
    alive = np.array([True, True])
    assert np.allclose(SpeciesGrowth(g, pref, Rs, alive), target)

    target = targets[1]
    g = np.array([[1.0, 0.9], [0.8, 1.1]])
    pref = np.array([[1, 2], [2, 1]])
    Rs = np.array([1.0, 1.0])
    alive = np.array([True, False])
    assert np.allclose(SpeciesGrowth(g, pref, Rs, alive), target)

    target = targets[2]
    g = np.array([[1.0, 0.9, 0.7], [0.8, 1.1, 0.2], [0.3, 1.5, 0.6]])
    pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    Rs = np.array([1.0, 0, 0])
    alive = np.array([True, True, True])
    assert np.allclose(SpeciesGrowth(g, pref, Rs, alive), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e