from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial



def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calculates the species growth rate
    Inputs:
    spc: current species abundance, 1D array of length N
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''
    
    # Precompute the product of resources and their efficiencies
    weighted_resources = [res[j] * w[j] for j in range(len(res))]
    
    # Initialize growth rate array using a dictionary for an alternative approach
    effective_intake_dict = {i: 0 for i in range(len(b))}
    
    # Fill the dictionary with effective resource intake for each species
    for i in range(len(b)):
        for j in range(len(res)):
            effective_intake_dict[i] += c[i][j] * weighted_resources[j]
    
    # Calculate growth rate for each species using a dictionary lookup
    g_spc = []
    for i in range(len(b)):
        growth_rate = b[i] * (effective_intake_dict[i] - m[i])
        g_spc.append(growth_rate)
    
    return g_spc


try:
    targets = process_hdf5_to_tuple('25.1', 3)
    target = targets[0]
    spc = np.array([1, 1])
    res = np.array([0.2, 0.4])
    b = np.array([1, 1])
    c = np.array([[0.4, 0.5], [0.9, 0.1]])
    w = np.array([0.8, 0.4])
    m = np.array([0.1, 0.05])
    assert np.allclose(SpeciesGrowth(spc, res, b, c, w, m), target)

    target = targets[1]
    spc = np.array([0.5, 0.6, 0.7])
    res = np.array([0.0, 0.0])
    b = np.array([0.1, 0.1, 0.1])
    c = np.array([[1, 0.1], [0.1, 1], [0.2, 0.3]])
    w = np.array([1, 1])
    m = np.array([0.05, 0.02, 0.1])
    assert np.allclose(SpeciesGrowth(spc, res, b, c, w, m), target)

    target = targets[2]
    spc = np.array([0.5, 0.6, 0.7])
    res = np.array([0.2, 0.4])
    b = np.array([0.1, 0.1, 0.1])
    c = np.array([[1, 0.1], [0.1, 1], [0.2, 0.3]])
    w = np.array([1, 1])
    m = np.array([0.05, 0.02, 0.1])
    assert np.allclose(SpeciesGrowth(spc, res, b, c, w, m), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e