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
    g_temp = np.zeros(N, dtype=float)
    r_temp = np.zeros(N, dtype=int)

    for i in range(N):
        if alive[i]:
            for j in range(R):
                resource_index = pref[i, j] - 1  # Convert to 0-based index
                if Rs[resource_index] > 0:
                    # Calculate the growth rate for the species i using its most preferred available resource
                    g_temp[i] = g[i, resource_index]
                    r_temp[i] = resource_index + 1  # Convert back to 1-based index
                    Rs[resource_index] -= 1  # Decrement the resource level
                    break
        else:
            g_temp[i] = 0
            r_temp[i] = 0
    
    return g_temp, r_temp



def OneCycle(g, pref, spc_init, Rs, T):
    '''This function simulates the dynamics in one dilution cycle. 
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    spc_init: species abundance at the beginning of cycle. 1d float numpy array of length N. 
    Rs: resource level in environment at the beginning of cycle. 1d float numpy array of length R. 
    T: time span of dilution cycle. float. 
    Outputs: 
    spc_end: species abundance at the end of cycle. 1d float numpy array of length N. 
    Rs_end: resource level in environment at the end of cycle. 1d float numpy array of length R.
    '''
    
    N, R = g.shape
    spc_current = np.copy(spc_init)
    Rs_current = np.copy(Rs)
    
    # While there are resources available, let the species grow
    while np.any(Rs_current > 0):
        g_temp, r_temp = SpeciesGrowth(g, pref, Rs_current, spc_current > 0)
        
        # Species grow exponentially based on the growth rate and time
        spc_growth = spc_current * np.exp(g_temp * T)
        
        # Calculate resource consumption
        for i in range(N):
            if r_temp[i] > 0:
                resource_index = r_temp[i] - 1
                Rs_current[resource_index] = max(0, Rs_current[resource_index] - spc_growth[i])

        # Update species abundance
        spc_current = spc_growth

    # At the end of the cycle, apply the dilution factor
    dilution_factor = 0.1
    spc_end = spc_current * dilution_factor
    Rs_end = Rs_current  # Resources remain as they are at the end of the cycle
    
    return spc_end, Rs_end


try:
    targets = process_hdf5_to_tuple('26.2', 3)
    target = targets[0]
    g = np.array([[1.0, 0.9, 0.7], [0.8, 1.1, 0.2], [0.3, 1.5, 0.6]])
    pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    spc_init = np.array([0.01, 0.02, 0.03])
    Rs = np.array([1.0, 1.0, 1.0])
    T = 24
    assert np.allclose(OneCycle(g, pref, spc_init, Rs, T), target)

    target = targets[1]
    g = np.array([[1.0, 0.9, 0.7], [0.8, 1.1, 0.2], [0.3, 1.5, 0.6]])
    pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    spc_init = np.array([0.0, 0.02, 0.03])
    Rs = np.array([1.0, 1.0, 1.0])
    T = 24
    assert np.allclose(OneCycle(g, pref, spc_init, Rs, T), target)

    target = targets[2]
    g = np.array([[1.0, 0.9, 0.7], [0.8, 1.1, 0.2], [0.3, 1.5, 0.6]])
    pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    spc_init = np.array([0.0, 0.02, 0.03])
    Rs = np.array([1.0, 1.0, 1.0])
    T = 2
    assert np.allclose(OneCycle(g, pref, spc_init, Rs, T), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e