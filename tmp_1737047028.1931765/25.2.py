import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

# Background: The MacArthur consumer-resource model describes the growth rate of species based on their interaction with resources. 
# The growth rate for a species i, denoted as g_i, is calculated using the formula:
# g_i = b_i * (sum over beta of (c_{iβ} * w_β * R_β) - m_i)
# where:
# - b_i is the inverse timescale of species dynamics, representing how quickly the species can respond to changes.
# - c_{iβ} is the consumer-resource conversion matrix, indicating how effectively species i can convert resource β into growth.
# - w_β is the efficiency or value of resource β.
# - R_β is the current level of resource β.
# - m_i is the maintenance cost for species i, representing the baseline resource requirement for survival.
# The growth rate g_i is positive if the resources available, weighted by their efficiency and conversion rates, exceed the maintenance cost.

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



# Background: In ecological models, resources are not only consumed by species but also have their own dynamics. 
# The logistic growth model is often used to describe the natural replenishment of resources. 
# The logistic growth rate of a resource is determined by its current abundance, its intrinsic growth rate, 
# and its carrying capacity. The carrying capacity is the maximum population size of the resource that the environment can sustain indefinitely.
# The change in resource level due to consumption by species is determined by the consumer-resource conversion matrix and the species abundance.
# The net change in resource level is the difference between its natural logistic growth and the consumption by species.


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
    consumption = np.dot(spc, c)
    
    # Calculate the net change in resources
    f_res = logistic_growth - consumption
    
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
