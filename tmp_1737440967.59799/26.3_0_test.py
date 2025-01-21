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


# Background: In a biological system, species grow exponentially under ideal conditions without resource constraints. 
# The exponential growth can be described by the equation N(t) = N0 * exp(r * t), where N0 is the initial abundance, 
# r is the growth rate, and t is the time. In ecology, resources can be consumed by species during their growth, 
# leading to a change in resource levels over time. This simulation involves calculating the species' abundance 
# and the remaining resource levels after a given time period (cycle) based on their exponential growth and resource consumption.


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
    
    # Initialize the output arrays for species abundance and resource levels at the end of the cycle
    spc_end = np.zeros(N, dtype=float)
    Rs_end = np.copy(Rs)
    
    # Calculate the growth rate and resource consumption for each species
    g_temp, r_temp = SpeciesGrowth(g, pref, Rs, spc_init > 0)
    
    # Simulate the growth of each species and resource consumption over the cycle
    for i in range(N):
        if spc_init[i] > 0:  # Check if species is present
            # Calculate the final abundance after exponential growth
            spc_end[i] = spc_init[i] * np.exp(g_temp[i] * T)
            
            # Determine the resource being consumed by the species
            resource_index = r_temp[i] - 1  # Convert 1-based index to 0-based index
            
            if resource_index >= 0:
                # Calculate the resource consumption
                resource_consumption = spc_end[i] - spc_init[i]
                Rs_end[resource_index] = max(0, Rs_end[resource_index] - resource_consumption)
    
    return spc_end, Rs_end



# Background: In ecological systems, dilution cycles are used to mimic natural processes where resources and populations 
# might be periodically refreshed. A dilution cycle involves reducing the concentrations of all components (both species and 
# resources) by a certain factor, simulating a natural or experimental 'reset' to a new baseline. After dilution, the populations 
# and resources are moved to fresh media where a new supply of resources is introduced, akin to seasonal or regular resource 
# influxes in nature. Over many cycles, species with insufficient growth rates or resource access may fall below a critical 
# threshold and become extinct. The aim is to simulate multiple such cycles and determine which species survive the iterative 
# dilution and resource replenishment processes.

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


    # Initialize species abundance and resource levels
    spc_current = np.copy(spc_init)
    Rs_current = np.copy(Rs)

    for cycle in range(N_cycles):
        # Simulate one cycle
        spc_end, Rs_end = OneCycle(g, pref, spc_current, Rs_current, T)

        # Apply dilution to species and resources
        spc_current = spc_end / D
        Rs_current = (Rs_end / D) + Rs  # Replenish resources with a fresh chunk

        # Apply extinction threshold
        spc_current[spc_current < SPC_THRES] = 0

    # Determine surviving species after all cycles
    survivors = [idx for idx, abundance in enumerate(spc_current) if abundance > 0]

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
