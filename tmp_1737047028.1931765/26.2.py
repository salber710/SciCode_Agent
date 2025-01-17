import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy

# Background: In ecology, the growth rate of a species is influenced by the availability of resources and the species' preference for those resources. Each species has a set of preferred resources, and its growth rate is determined by the availability of these resources in the environment. If a species is present, it will consume its most preferred available resource, which contributes to its growth rate. The task is to calculate the current growth rate of each species based on the resources they consume and to identify which resource each species is consuming. If a species is not present or no resources are available, its growth rate is zero, and it consumes no resources.


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
    
    for i in range(N):
        if alive[i]:
            for j in range(R):
                resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
                if Rs[resource_index] > 0:
                    g_temp[i] = g[i, resource_index]
                    r_temp[i] = resource_index + 1  # Store as 1-based index
                    break
    
    return g_temp, r_temp



# Background: In ecological modeling, species abundance and resource levels can change over time due to growth and consumption. 
# During a dilution cycle, species grow exponentially based on available resources, and resources are consumed by the species. 
# The growth of a species is determined by its growth rate on the resource it consumes. The exponential growth model is given by 
# the formula N(t) = N0 * exp(r * t), where N0 is the initial abundance, r is the growth rate, and t is the time. 
# The resource consumption reduces the available resources, which affects the growth rates in subsequent cycles.


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
    
    # Determine the growth rate and resource consumption for each species
    for i in range(N):
        for j in range(R):
            resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
            if Rs_end[resource_index] > 0:
                growth_rate = g[i, resource_index]
                # Update species abundance using exponential growth
                spc_end[i] = spc_init[i] * np.exp(growth_rate * T)
                # Assume each species consumes a fixed amount of resource proportional to its growth
                # Here, we assume a simple model where consumption is proportional to growth
                Rs_end[resource_index] -= spc_end[i] - spc_init[i]
                # Ensure resources do not go negative
                Rs_end[resource_index] = max(Rs_end[resource_index], 0)
                break
    
    return spc_end, Rs_end


from scicode.parse.parse import process_hdf5_to_tuple

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
