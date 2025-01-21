import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy

# Background: In ecology, the growth rate of a species in an environment is influenced by the availability of resources and the preference of the species for these resources. Each species has a growth matrix that defines its growth rate on different resources. The preference order array indicates the preferred resources for each species. The goal is to calculate the growth rate of each species based on available resources and whether the species is present in the environment. The algorithm should consider each species' preference order and current resource levels to determine which resource each species consumes, updating the growth rate accordingly.


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
    
    N = len(alive)  # Number of species
    R = len(Rs)     # Number of resources
    
    # Initialize the output arrays
    g_temp = np.zeros(N, dtype=float)
    r_temp = np.zeros(N, dtype=int)
    
    for i in range(N):
        if alive[i]:  # Check if species i is present
            # Check the species' preference list for available resources
            for j in range(R):
                resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based index
                if Rs[resource_index] > 0:  # Check if the resource is available
                    # Update growth rate and resource consumption
                    g_temp[i] = g[i, resource_index]
                    r_temp[i] = resource_index + 1  # Convert back to 1-based index for output
                    break  # Stop after finding the first available preferred resource
    
    return g_temp, r_temp



# Background: In ecology, a dilution cycle represents a period during which species grow and resources are utilized. 
# During a cycle, species grow exponentially based on available resources, and resource levels decrease as they are consumed.
# The exponential growth of a species is determined by the formula: N(t) = N0 * exp(growth_rate * t), where N0 is the initial 
# abundance, 't' is the time period, and 'growth_rate' is the growth rate of the species. At the end of the cycle, 
# the abundance of each species and the remaining resources need to be calculated. 


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
    
    N = len(spc_init)  # Number of species
    R = len(Rs)        # Number of resources
    
    # Initialize species abundance and resource levels at the end of the cycle
    spc_end = np.copy(spc_init)
    Rs_end = np.copy(Rs)
    
    # Determine which resources each species is consuming and calculate growth rates
    growth_rates, r_temp = SpeciesGrowth(g, pref, Rs, spc_init > 0)
    
    # Update species abundance based on exponential growth
    for i in range(N):
        if spc_init[i] > 0:  # If species is present
            spc_end[i] = spc_init[i] * np.exp(growth_rates[i] * T)
            resource_index = r_temp[i] - 1  # Convert to 0-based index
            if resource_index >= 0:
                # Assuming each species consumes a fixed amount of resource proportional to its growth
                Rs_end[resource_index] -= (spc_end[i] - spc_init[i]) / g[i, resource_index]
                Rs_end[resource_index] = max(0, Rs_end[resource_index])  # Resource level cannot be negative
    
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
