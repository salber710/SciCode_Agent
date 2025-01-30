from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy


def SpeciesGrowth(g, pref, Rs, alive):
    N, R = g.shape
    g_temp = np.zeros(N, dtype=float)
    r_temp = np.zeros(N, dtype=int)

    # Create a list of tuples (growth potential, resource index) for each species
    growth_potentials = [[] for _ in range(N)]

    for i in range(N):
        if alive[i]:
            # Populate growth potentials based on resource preferences and availability
            for j in range(R):
                resource_index = pref[i, j] - 1
                if Rs[resource_index] > 0:
                    growth_potentials[i].append((g[i, resource_index], resource_index))

            # Sort by growth potential descending, then by preference order
            growth_potentials[i].sort(key=lambda x: (-x[0], pref[i].tolist().index(x[1] + 1)))

            if growth_potentials[i]:
                best_growth, chosen_resource = growth_potentials[i][0]
                g_temp[i] = best_growth
                r_temp[i] = chosen_resource + 1
                Rs[chosen_resource] -= 1

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
    spc_end = np.zeros(N, dtype=float)
    Rs_end = np.copy(Rs)
    
    # A matrix to track the growth contribution for each species
    growth_contribution = np.zeros((N, R), dtype=float)
    
    # Calculate the growth contribution for each species-resource pair
    for i in range(N):
        for j in range(R):
            resource_index = pref[i, j] - 1
            if Rs_end[resource_index] > 0:
                # Contribution is proportional to the growth rate and resource availability
                growth_contribution[i, resource_index] = g[i, resource_index] * Rs_end[resource_index]
    
    # Normalize the growth contributions for each species
    for i in range(N):
        total_contribution = np.sum(growth_contribution[i, :])
        if total_contribution > 0:
            growth_contribution[i, :] /= total_contribution
    
    # Calculate the effective growth rate for each species
    effective_growth_rate = np.zeros(N, dtype=float)
    for i in range(N):
        effective_growth_rate[i] = np.dot(growth_contribution[i, :], g[i, :])
    
    # Update the species abundance
    for i in range(N):
        spc_end[i] = spc_init[i] * np.exp(effective_growth_rate[i] * T)
    
    # Update the resource levels
    resource_usage = np.sum(growth_contribution, axis=0)
    Rs_end = np.maximum(Rs_end - resource_usage, 0)
    
    return spc_end, Rs_end




def SimulatedCycles(g, pref, spc_init, Rs, SPC_THRES, T, D, N_cycles):
    '''
    This function simulates multiple dilution cycles and returns the survivors.
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
    species_abundance = np.array(spc_init, copy=True)
    current_resources = np.array(Rs, copy=True)

    # Simulate the dilution cycles
    for cycle in range(N_cycles):
        # Track the consumption and growth in a dictionary for clarity
        growth_contributions = {i: 0 for i in range(len(species_abundance))}

        # Iterate over each species
        for i in range(len(species_abundance)):
            # Calculate growth based on species' preference order
            for rank in range(len(current_resources)):
                resource_idx = np.where(pref[i] == rank + 1)[0][0]
                if current_resources[resource_idx] > 0:
                    # Compute growth contribution
                    growth_contrib = g[i, resource_idx] * current_resources[resource_idx] * T
                    growth_contributions[i] += growth_contrib
                    # Update resource level
                    current_resources[resource_idx] = max(0, current_resources[resource_idx] - growth_contrib)

        # Update species abundances with exponential growth
        for i in range(len(species_abundance)):
            species_abundance[i] *= np.exp(growth_contributions[i])

        # Apply dilution to species abundances and resources
        species_abundance *= D
        current_resources *= D

        # Refresh resources to their initial levels
        current_resources = np.clip(current_resources + Rs, 0, Rs)

        # Set species below extinction threshold to zero
        species_abundance[species_abundance < SPC_THRES] = 0

    # Determine surviving species
    survivors = [idx for idx, abundance in enumerate(species_abundance) if abundance > 0]

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