import numpy as np
from math import *
from scipy.optimize import root_scalar
from scipy import special
import copy



# Background: In ecology, the growth rate of a species is influenced by the availability of resources and the species' preference for those resources. Each species has a set of preferred resources, and its growth rate is determined by the availability of these resources in the environment. If a species is present, it will consume its most preferred available resource, which contributes to its growth rate. The task is to calculate the current growth rate of each species based on the resources they consume and to identify which resource each species is consuming. If a species is not present or no resources are available, its growth rate is zero, and it consumes no resources.


def SpeciesGrowth(g, pref, Rs, alive):
    '''This function calculates the species growth rate
    Inputs:
    g: growth matrix of species i on resource j. 2d float numpy array of size (N, R). 
    pref: species' preference order, 2d numpy array with dimensions [N, R] and int elements between 1 and R
    Rs: resource level in environment. 1d float numpy array of length R. 
    alive: whether the species is present or not. 1d boolean numpy array of length N. 
    Outputs: 
    g_temp: current growth rate of species, 1D float numpy array of length N. 
    r_temp: list of resources that each species is eating. 1D int numpy array of length N. 
    '''
    
    N, R = g.shape
    g_temp = np.zeros(N)
    r_temp = np.zeros(N, dtype=int)
    
    for i in range(N):
        if alive[i]:
            for j in range(R):
                resource_index = pref[i, j] - 1  # Convert 1-based index to 0-based
                if Rs[resource_index] > 0:
                    g_temp[i] = g[i, resource_index]
                    r_temp[i] = resource_index + 1  # Store as 1-based index
                    break
    
    return g_temp, r_temp


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('26.1', 3)
target = targets[0]

g = np.array([[1.0, 0.9], [0.8, 1.1]])
pref = np.array([[1, 2], [2, 1]])
Rs = np.array([0, 0])
alive = np.array([True, True])
assert np.allclose(SpeciesGrowth(g, pref, Rs, alive), target)
target = targets[1]

g = np.array([[1.0, 0.9], [0.8, 1.1]])
pref = np.array([[1, 2], [2, 1]])
Rs = np.array([1.0, 1.0])
alive = np.array([True, False])
assert np.allclose(SpeciesGrowth(g, pref, Rs, alive), target)
target = targets[2]

g = np.array([[1.0, 0.9, 0.7], [0.8, 1.1, 0.2], [0.3, 1.5, 0.6]])
pref = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
Rs = np.array([1.0, 0, 0])
alive = np.array([True, True, True])
assert np.allclose(SpeciesGrowth(g, pref, Rs, alive), target)
