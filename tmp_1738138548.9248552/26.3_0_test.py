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



def OneCycle(g, pref, spc_init, Rs, T):
    N, R = g.shape
    spc_end = np.copy(spc_init)
    Rs_end = np.copy(Rs)
    
    # Calculate growth rates using a dynamic resource allocation based on exponential decay of preference
    for i in range(N):
        resource_utilization = np.zeros(R)
        for j in range(R):
            # Calculate the decay factor for resource preference
            decay_factor = np.exp(-pref[i, j])
            resource_utilization[j] = decay_factor * g[i, j] * Rs[j]
        
        # Total growth rate is the sum of all utilization factors
        total_growth_rate = np.sum(resource_utilization)
        
        # Update species abundance
        spc_end[i] = spc_init[i] * np.exp(total_growth_rate * T)
        
        # Update resource levels based on the utilized amounts
        for j in range(R):
            Rs_end[j] -= resource_utilization[j] * T
            if Rs_end[j] < 0:
                Rs_end[j] = 0

    return spc_end, Rs_end



# Background: In ecological modeling, dilution cycles are used to simulate the periodic renewal of resources and the removal of biomass, mimicking natural processes such as tides or seasonal changes. During each cycle, the abundance of species and the levels of resources are reduced by a dilution factor, and then replenished with a fresh supply of resources. This process can affect the survival of species, as those with abundances falling below a certain threshold are considered extinct. The goal is to simulate multiple cycles and determine which species survive after all cycles, given their growth rates, resource preferences, and initial conditions.


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
    
    # Initial resources to be added every cycle
    initial_resources = np.copy(Rs)
    
    # Current species abundance and resource levels
    spc_current = np.copy(spc_init)
    Rs_current = np.copy(Rs)
    
    for cycle in range(N_cycles):
        # Simulate one cycle
        spc_current, Rs_current = OneCycle(g, pref, spc_current, Rs_current, T)
        
        # Apply dilution to species and resources
        spc_current *= D
        Rs_current *= D
        
        # Add fresh resources
        Rs_current += initial_resources
        
        # Check for extinction
        spc_current[spc_current < SPC_THRES] = 0
    
    # Determine surviving species
    survivors = [idx for idx, abundance in enumerate(spc_current) if abundance > 0]
    
    return survivors


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e