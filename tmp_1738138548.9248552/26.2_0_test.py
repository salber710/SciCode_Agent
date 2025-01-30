from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy


def SpeciesGrowth(g, pref, Rs, alive):
    N, R = g.shape
    g_temp = np.zeros(N)
    r_temp = np.zeros(N, dtype=int)
    
    # Calculate the efficiency of each resource for each species
    efficiency = g / np.maximum(Rs, 1e-10)  # Avoid division by zero
    
    for i in range(N):
        if alive[i]:
            # Find the resource with the highest efficiency that is preferred by the species
            best_efficiency = -np.inf
            chosen_resource = 0
            for j in range(R):
                resource_index = pref[i, j] - 1
                if Rs[resource_index] > 0 and efficiency[i, resource_index] > best_efficiency:
                    best_efficiency = efficiency[i, resource_index]
                    chosen_resource = resource_index
            
            if chosen_resource != 0 or best_efficiency != -np.inf:
                g_temp[i] = g[i, chosen_resource]
                r_temp[i] = chosen_resource + 1  # Store as 1-based index
                # Decrease the resource level to simulate consumption
                Rs[chosen_resource] -= 1  # This can be adjusted based on consumption rate
    
    return g_temp, r_temp



# Background: In ecological modeling, species growth can often be modeled using exponential growth equations, 
# especially over short time spans where resources are not limiting. The exponential growth model is given by 
# the equation N(t) = N0 * exp(r * t), where N0 is the initial population size, r is the growth rate, and t is 
# the time. In this context, we simulate a dilution cycle where species grow exponentially based on their 
# growth rates on available resources, and resources are consumed by the species. The goal is to calculate 
# the species abundance and resource levels at the end of the cycle.


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
    spc_end = np.copy(spc_init)
    Rs_end = np.copy(Rs)
    
    # Calculate the growth rate for each species based on available resources
    for i in range(N):
        # Find the most preferred available resource
        for j in range(R):
            resource_index = pref[i, j] - 1
            if Rs_end[resource_index] > 0:
                # Calculate the growth of species i using the chosen resource
                growth_rate = g[i, resource_index]
                spc_end[i] = spc_init[i] * np.exp(growth_rate * T)
                # Simulate resource consumption
                Rs_end[resource_index] -= spc_end[i] - spc_init[i]
                if Rs_end[resource_index] < 0:
                    Rs_end[resource_index] = 0
                break
    
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