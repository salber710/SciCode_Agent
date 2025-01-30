from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial



# Background: The MacArthur consumer-resource model describes the growth rate of species based on their interaction with available resources. 
# The growth rate for a species is determined by the balance between the benefits it gains from consuming resources and the costs it incurs 
# for maintenance. The growth rate g_i for species i is calculated using the formula:
# g_i = b_i * (sum over beta of (c_{iβ} * w_β * R_β) - m_i)
# where:
# - b_i is the inverse timescale of species dynamics, representing how quickly the species can respond to changes.
# - c_{iβ} is the conversion efficiency of resource β into biomass for species i.
# - w_β is the efficiency or value of resource β.
# - R_β is the current level of resource β.
# - m_i is the maintenance cost for species i.
# The sum over β represents the total benefit a species gains from all resources, and subtracting the maintenance cost gives the net growth rate.

def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calculates the species growth rate
    Inputs:
    spc: current species abundance, 1D array of length N (not used in this calculation)
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''
    # Calculate the effective resource contribution for each species
    effective_resource_contribution = np.dot(c, w * res)
    
    # Calculate the growth rate for each species
    g_spc = b * (effective_resource_contribution - m)
    
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