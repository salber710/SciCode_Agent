from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial


def SpeciesGrowth(spc, res, b, c, w, m):
    # Calculate the total resource utilization by each species
    resource_utilization = np.sum(c * (res[:, None] * w), axis=0)
    
    # Compute the growth rates by subtracting maintenance costs and scaling by the inverse timescale
    g_spc = b * (resource_utilization - m)
    
    return g_spc



# Background: In ecological modeling, resources consumed by species often follow logistic growth dynamics. 
# Logistic growth is characterized by an initial exponential growth when resources are abundant, 
# followed by a slowdown as resources approach a carrying capacity, K. The rate of resource replenishment 
# is influenced by the inverse timescale of growth, r, which determines how quickly resources can recover. 
# The consumption of resources by species is determined by the conversion matrix, c, which indicates 
# how much of each resource is consumed by each species. The net change in resource levels is the 
# difference between the logistic growth and the consumption by species.

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
    # This is the sum of the product of species abundance and their conversion rates for each resource
    resource_consumption = np.sum(spc[:, None] * c, axis=0)
    
    # The net change in resources is the logistic growth minus the consumption
    f_res = logistic_growth - resource_consumption
    
    return f_res


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e