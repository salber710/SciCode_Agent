import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy

# Background: In ecological modeling, the growth rate of a species is often influenced by the availability of resources and the species' preference for those resources. Each species has a set of preferred resources, and its growth rate is determined by the availability of these resources in the environment. The growth matrix `g` indicates how well each species grows on each resource. The preference order `pref` specifies which resources a species prefers, ranked from most to least preferred. The resource levels `Rs` indicate the current availability of each resource. The `alive` array indicates whether a species is present in the environment. The task is to calculate the current growth rate of each species based on the resources they can consume and their preferences, and to determine which resource each species is currently consuming.


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
            found_resource = False
            for j in range(R):
                resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
                if resource_index >= 0 and resource_index < R and Rs[resource_index] > 0:
                    g_temp[i] = g[i, resource_index]
                    r_temp[i] = resource_index + 1  # Store as 1-based index
                    found_resource = True
                    break
            if not found_resource:
                r_temp[i] = 0  # No available resource found
        else:
            r_temp[i] = 0  # Species is not alive, hence no resource is being consumed
    
    return g_temp, r_temp



# Background: In ecological modeling, species growth can be modeled using exponential growth equations. 
# The abundance of a species at the end of a time period can be calculated using the formula:
# N(t) = N(0) * exp(r * t), where N(0) is the initial abundance, r is the growth rate, and t is the time period.
# During a dilution cycle, species consume resources, which affects their growth rates. 
# The growth rates are determined by the availability of resources and the species' preferences for those resources.
# The task is to simulate one cycle of growth and resource consumption, updating both species abundances and resource levels.


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
        # Find the preferred resource that is available
        for j in range(R):
            resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
            if Rs_end[resource_index] > 0:
                # Calculate the growth rate for species i using the available resource
                growth_rate = g[i, resource_index]
                # Update species abundance using exponential growth formula
                spc_end[i] = spc_init[i] * np.exp(growth_rate * T)
                # Assume resource consumption is proportional to growth
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
