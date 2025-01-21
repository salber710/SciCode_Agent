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



# Background: In ecology, the dynamics of species populations and resource levels can be modeled using exponential growth functions over a given time period. The abundance of each species at the end of a cycle can be determined by its initial abundance and its growth rate, assuming exponential growth. The growth rate for each species is determined by the available resources and the species' preference for these resources. As resources are consumed by species, their levels decrease accordingly. The task is to simulate one cycle of these dynamics, updating species abundance and resource levels.


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

    # Determine growth rates for each species based on available resources
    g_temp = np.zeros(N, dtype=float)
    r_temp = np.zeros(N, dtype=int)

    for i in range(N):
        # Check the species' preference list for available resources
        for j in range(R):
            resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based index
            if Rs[resource_index] > 0:  # Check if the resource is available
                # Update growth rate and resource consumption
                g_temp[i] = g[i, resource_index]
                r_temp[i] = resource_index
                break  # Stop after finding the first available preferred resource

    # Calculate species abundance at the end of the cycle
    spc_end = spc_init * np.exp(g_temp * T)

    # Update resource levels
    Rs_end = Rs.copy()
    for i in range(N):
        if r_temp[i] != 0:  # If species is consuming a resource
            Rs_end[r_temp[i]] -= spc_end[i] - spc_init[i]  # Assumes each unit of growth consumes one unit of resource

    # Ensure no negative resources due to overconsumption
    Rs_end = np.maximum(Rs_end, 0)

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
