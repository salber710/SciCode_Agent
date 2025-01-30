from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial

def SpeciesGrowth(spc, res, b, c, w, m):
    '''This function calculates the species growth rate
    Inputs:
    res: resource level, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    Outputs: 
    g_spc: growth rate of species, 1D array of length N
    '''
    
    # Calculate the growth rate for each species
    g_spc = np.zeros_like(b)
    for i in range(len(spc)):
        # Calculate the term inside the sum for species i
        resource_contribution = np.sum(c[i, :] * w * res)
        # Calculate the growth rate for species i
        g_spc[i] = b[i] * (resource_contribution - m[i])
    
    return g_spc



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
    
    # Initialize the resource growth rate array
    f_res = np.zeros_like(res)
    
    for alpha in range(len(res)):
        # Logistic growth of the resource
        logistic_growth = r[alpha] * (K[alpha] - res[alpha]) * res[alpha] / K[alpha]
        
        # Consumption by species
        consumption = np.sum(spc * c[:, alpha] * res[alpha])
        
        # Rate of change of each resource
        f_res[alpha] = logistic_growth - consumption
    
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