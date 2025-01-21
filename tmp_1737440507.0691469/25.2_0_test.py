import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

# Background: The MacArthur model is a classic model in theoretical ecology that describes the growth rate of a species in an ecosystem. 
# The growth rate, g_i, of a species i is determined by the balance between the resources it can convert into biomass and its maintenance costs.
# The term b_i is the inverse timescale, reflecting how quickly the species can grow. The sum over β represents the total contribution 
# of all resources to the growth of species i, where c_{iβ} is the conversion efficiency of species i for resource β, w_β is the value or 
# efficiency of resource β, and R_β is the current level of resource β. The term m_i is the maintenance cost for species i, which must 
# be subtracted from its potential growth. The goal is to compute the growth rates for all species given these parameters.


def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calculates the species growth rate
    Inputs:
    spc: species abundance, 1D array of length N (not directly used in this function)
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''
    
    # Calculate the contribution of resources to the growth of each species
    resource_contribution = np.dot(c, w * res)
    
    # Calculate the growth rate for each species
    g_spc = b * (resource_contribution - m)
    
    return g_spc



# Background: In ecosystems, resources such as nutrients or energy sources are not only consumed by species but also naturally replenish themselves over time. This replenishment can often be modeled using logistic growth, which describes how resources grow rapidly when they are scarce and slow down as they approach a carrying capacity, K. The growth rate of each resource is influenced by its current abundance and its carrying capacity. Additionally, resources are depleted by consumption from species, which is determined by the species' abundance and conversion efficiency. The logistic growth is characterized by the inverse timescale r, which governs how quickly each resource can grow towards its carrying capacity.


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

    # Calculate logistic growth component for each resource
    logistic_growth = r * res * (1 - res / K)
    
    # Calculate resource consumption by species
    consumption = np.dot(spc, c)
    
    # Calculate net change in resource levels
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
