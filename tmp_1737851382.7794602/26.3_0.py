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
                consumed_resource = (spc_end[i] - spc_init[i]) if spc_end[i] > spc_init[i] else 0
                Rs_end[resource_index] -= consumed_resource
                # Ensure resources do not go negative
                Rs_end[resource_index] = max(Rs_end[resource_index], 0)
                break
    
    return spc_end, Rs_end



# Background: In ecological modeling, simulating multiple dilution cycles involves iterating over a series of growth and resource consumption phases. 
# Each cycle consists of species growing based on available resources, followed by a dilution step where both species and resources are reduced by a factor D. 
# After dilution, a fresh supply of resources is added to the environment. The process is repeated for a specified number of cycles. 
# Species whose abundance falls below a certain threshold (SPC_THRES) are considered extinct and are not included in the list of survivors. 
# The goal is to determine which species survive after all cycles have been completed.


def SimulatedCycles(g, pref, spc_init, Rs, SPC_THRES, T, D, N_cycles):
    '''This function simulates multiple dilution cycles and return the survivors
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    spc_init: species abundance at the beginning of cycle. 1d float numpy array of length N. 
    Rs: resource level in environment at the beginning of cycle. 1d float numpy array of length R. 
    SPC_THRES: species dieout cutoff, float
    T: time span of dilution cycle. float. 
    D: dilution rate, float
    N_cycles: number of dilution cycles, int. 
    Outputs: 
    survivors: list of surviving species, elements are integers
    '''
    
    N, R = g.shape
    spc_current = np.copy(spc_init)
    Rs_initial = np.copy(Rs)
    
    for cycle in range(N_cycles):
        # Simulate one cycle
        spc_current, Rs_current = OneCycle(g, pref, spc_current, Rs, T)
        
        # Dilute species and resources
        spc_current *= D
        Rs_current *= D
        
        # Add fresh resources
        Rs_current += Rs_initial
        
        # Update Rs for the next cycle
        Rs = Rs_current
    
    # Determine surviving species
    survivors = [idx for idx, abundance in enumerate(spc_current) if abundance > SPC_THRES]
    
    return survivors

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('26.3', 3)
target = targets[0]

g = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
spc_init = np.array([0.01, 0.02, 0.03])
Rs = np.array([1.0, 1.0, 1.0])
SPC_THRES = 1e-7
T = 24
D = 100
N_cycles = 1000
assert np.allclose(SimulatedCycles(g, pref, spc_init, Rs, SPC_THRES, T, D, N_cycles), target)
target = targets[1]

g = np.array([[0.9, 0.1, 0.7], [0.8, 1.0, 0.2], [0.3, 1.3, 1.5]])
pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
spc_init = np.array([0.01, 0.02, 0.03])
Rs = np.array([1.0, 1.0, 1.0])
SPC_THRES = 1e-7
T = 24
D = 100
N_cycles = 1000
assert np.allclose(SimulatedCycles(g, pref, spc_init, Rs, SPC_THRES, T, D, N_cycles), target)
target = targets[2]

g = np.array([[1.0, 0.6, 0.9, 0.1], 
              [0.31, 1.02, 0.81, 0.68],
              [0.82, 0.69, 1.03, 0.89], 
              [0.65, 0.44, 0.91, 1.01], 
              [0.9, 0.9, 0.89, 0.91]])
pref = np.argsort(-g, axis=1) + 1
spc_init = np.ones(5)*0.01
Rs= np.ones(4)
SPC_THRES = 1e-7
T = 24
D = 100
N_cycles = 1000
assert np.allclose(SimulatedCycles(g, pref, spc_init, Rs, SPC_THRES, T, D, N_cycles), target)
