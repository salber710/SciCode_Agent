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
    spc_end = np.copy(spc_init)
    Rs_end = np.copy(Rs)

    # Determine the contribution of each resource to each species' growth
    for i in range(N):
        growth_rate = 0
        # Check resources based on preference, accumulate the growth for all available resources
        for j in range(R):
            resource_index = pref[i, j] - 1
            if Rs_end[resource_index] > 0:
                # Use a fraction of the resource proportional to its availability and preference order
                resource_contribution = Rs_end[resource_index] / (j + 1)
                growth_rate += g[i, resource_index] * resource_contribution
                Rs_end[resource_index] -= resource_contribution

    # Calculate the final species abundance using accumulated growth rates over time T
    spc_end = spc_init * np.exp(growth_rate * T)

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