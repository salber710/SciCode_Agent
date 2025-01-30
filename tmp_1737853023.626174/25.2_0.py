import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

# Background: The MacArthur consumer-resource model describes the growth rate of species based on their interaction with available resources. 
# The growth rate for each species is determined by the balance between the benefits gained from consuming resources and the costs associated with maintaining the species. 
# The growth rate formula is given by: g_i = b_i * (sum(c_{iβ} * w_β * R_β) - m_i), where:
# - b_i is the inverse timescale of species dynamics, representing how quickly a species can respond to changes in resource levels.
# - c_{iβ} is the consumer-resource conversion matrix, indicating how effectively a species can convert resource β into its own biomass.
# - w_β is the efficiency or value of resource β.
# - R_β is the current level of resource β.
# - m_i is the maintenance cost for species i, representing the baseline resource requirement for survival.
# The goal is to compute the growth rate for each species based on these parameters.


def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calculates the species growth rate
    Inputs:
    spc: current species abundance, 1D array of length N (not used in the calculation directly)
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''
    # Calculate the effective resource consumption for each species
    effective_resource_consumption = np.dot(c, w * res)
    
    # Calculate the growth rate for each species
    g_spc = b * (effective_resource_consumption - m)
    
    return g_spc



# Background: In ecological models, resources consumed by species can also replenish themselves over time. 
# This replenishment is often modeled using logistic growth, which describes how resources grow rapidly when they are scarce 
# and slow down as they approach a carrying capacity. The logistic growth rate for a resource is given by 
# r_β * R_β * (1 - R_β / K_β), where:
# - r_β is the intrinsic growth rate of resource β, representing how quickly the resource can replenish itself.
# - R_β is the current level of resource β.
# - K_β is the carrying capacity of resource β, the maximum level the resource can reach.
# The consumption of resources by species is modeled by the term -sum(c_{iβ} * spc_i), which reduces the resource level 
# based on the abundance of species and their conversion efficiency. The goal is to compute the net change in resource levels 
# by combining these two effects.


def ResourcesUpdate(spc, res, c, r, K):
    '''This function calculates the changing rates of resources
    Inputs:
    spc: species population, 1D array of length N
    res: resource abundance, 1D array of length R
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    r: inverse timescale of resource growth, 1D array of length R
    K: resource carrying capacity, 1D array of length R
    Outputs: 
    f_res: growth rate of resources, 1D array of length R
    '''
    # Calculate the logistic growth component for each resource
    logistic_growth = r * res * (1 - res / K)
    
    # Calculate the consumption of resources by species
    resource_consumption = np.dot(spc, c)
    
    # Calculate the net change in resources
    f_res = logistic_growth - resource_consumption
    
    return f_res

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('25.2', 3)
target = targets[0]

spc = np.array([1, 1])
res = np.array([0.02, 0.04])
c = np.array([[1, 0.1], [0.1, 1]])
r = np.array([0.7, 0.9])
K = np.array([0.1, 0.05])
assert np.allclose(ResourcesUpdate(spc, res, c, r, K), target)
target = targets[1]

spc = np.array([0.05, 0.06, 0.07])
res = np.array([0.2, 0.4])
c = np.array([[1, 0.1], [0.1, 1], [0.2, 0.3]])
r = np.array([0.7, 0.9])
K = np.array([0.9, 0.5])
assert np.allclose(ResourcesUpdate(spc, res, c, r, K), target)
target = targets[2]

spc = np.array([0, 0, 0])
res = np.array([0.2, 0.4])
c = np.array([[1, 0.1], [0.1, 1], [0.2, 0.3]])
r = np.array([0.7, 0.7])
K = np.array([0.9, 0.5])
assert np.allclose(ResourcesUpdate(spc, res, c, r, K), target)
