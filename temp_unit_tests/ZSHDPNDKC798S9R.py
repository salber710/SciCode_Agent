
# Background: To simulate the dynamics of species and resources over time, we need to integrate the differential equations 
# governing their interactions. The species growth is determined by the MacArthur model, which considers the balance between 
# resource consumption and maintenance costs. Resources, on the other hand, follow logistic growth dynamics, replenishing 
# themselves while being consumed by species. The simulation will iterate over time from t=0 to t=tf with a timestep of dt, 
# updating species and resource levels at each step. Species with abundance below a specified threshold (SPC_THRES) at the 
# end of the simulation are considered extinct. The goal is to identify which species survive by the end of the simulation.

import numpy as np

def Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES):
    '''This function simulates the model's dynamics
    Inputs:
    spc_init: initial species population, 1D array of length N
    res_init: initial resource abundance, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resource, 1D array of length R
    m: species maintenance cost, 1D array of length N
    r: inverse timescale of resource growth, 1D array of length R
    K: resource carrying capacity, 1D array of length R
    tf: final time, float
    dt: timestep length, float
    SPC_THRES: species dieout cutoff, float
    Outputs: 
    survivors: list of integers (values between 0 and N-1)
    '''
    
    # Initialize species and resource levels
    spc = np.array(spc_init, dtype=float)
    res = np.array(res_init, dtype=float)
    
    # Time loop from 0 to tf with step dt
    t = 0
    while t < tf:
        # Calculate species growth rates
        effective_resource_consumption = np.dot(c, w * res)
        g_spc = b * (effective_resource_consumption - m)
        
        # Update species levels
        spc += g_spc * spc * dt
        
        # Calculate resource growth rates
        logistic_growth = r * res * (1 - res / K)
        resource_consumption = np.dot(spc, c)
        f_res = logistic_growth - resource_consumption
        
        # Update resource levels
        res += f_res * dt
        
        # Increment time
        t += dt
    
        # Check for negative or NaN values in species and resources to prevent simulation instability
        spc = np.clip(spc, 0, np.inf)
        res = np.clip(res, 0, np.inf)
    
    # Determine surviving species
    survivors = [i for i in range(len(spc)) if spc[i] >= SPC_THRES]
    
    return survivors
