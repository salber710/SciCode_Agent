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



# Background: In ecological models, resources such as nutrients or energy are consumed by species,
# which decreases their availability. However, resources can also regenerate over time. This regeneration
# is often modeled using logistic growth, which considers both the intrinsic growth rate of the resource
# and a carrying capacity that limits the maximum resource level. The rate of resource change is affected
# by both the consumption by species and the natural logistic growth. The inverse timescale of resource
# growth (r) determines how fast the resource can grow towards its carrying capacity (K). The conversion
# matrix (c) indicates how efficiently each species consumes each resource, which affects the resource 
# depletion rate based on species abundance.


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

    # Calculate the consumption of resources by species
    resource_consumption = np.dot(spc, c)

    # Calculate the logistic growth contribution
    logistic_growth = r * res * (1 - res / K)

    # The rate of change of resources is the logistic growth minus the consumption by species
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
