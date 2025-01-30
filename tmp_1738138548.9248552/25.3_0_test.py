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



def ResourcesUpdate(spc, res, c, r, K):
    # Calculate resource consumption using a geometric mean approach
    geometric_consumption = np.exp(np.dot(np.log(c + 1e-10), spc))
    
    # Calculate logistic growth using a modified logistic model incorporating a square root transformation
    modified_logistic_growth = r * np.sqrt(res) * (K - np.sqrt(res))
    
    # Compute the net change in resources
    f_res = modified_logistic_growth - geometric_consumption
    
    return f_res



# Background: In ecological modeling, simulating the dynamics of species and resources over time involves integrating differential equations that describe their interactions. The MacArthur model provides a framework for understanding how species grow based on resource consumption, while resources themselves replenish logistically. The simulation involves iteratively updating species and resource levels over discrete time steps until a final time is reached. The goal is to determine which species survive, based on whether their abundance remains above a certain threshold. This requires careful numerical integration and handling of the system's parameters.


def Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES):
    '''This function simulates the model's dynamics
    Inputs:
    spc_init: initial species population, 1D array of length N
    res_init: initial resource abundance, 1D array of length R
    b: inverse timescale of species dynamics, 1D array of length N
    c: consumer-resource conversion matrix, 2D array of shape [N, R]
    w: value/efficiency of resoruce, 1D array of length R
    m: species maintainance cost, 1D array of length N
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
    t = 0.0
    while t < tf:
        # Calculate growth rates for species
        resource_utilization = np.sum(c * (res[:, None] * w), axis=0)
        g_spc = b * (resource_utilization - m)
        
        # Update species abundance
        spc += g_spc * spc * dt
        
        # Calculate resource dynamics
        geometric_consumption = np.exp(np.dot(np.log(c + 1e-10), spc))
        modified_logistic_growth = r * np.sqrt(res) * (K - np.sqrt(res))
        f_res = modified_logistic_growth - geometric_consumption
        
        # Update resource levels
        res += f_res * dt
        
        # Ensure non-negative species and resources
        spc = np.maximum(spc, 0)
        res = np.maximum(res, 0)
        
        # Increment time
        t += dt
    
    # Determine surviving species
    survivors = [i for i in range(len(spc)) if spc[i] > SPC_THRES]
    
    return survivors


try:
    targets = process_hdf5_to_tuple('25.3', 3)
    target = targets[0]
    spc_init = np.array([0.5 for i in range(5)])
    res_init = np.array([0.1 for i in range(5)])
    b = np.array([0.9, 0.8, 1.1, 1.0, 1.0])*4
    c = np.eye(5)
    w = np.array([1 for i in range(5)])
    m = np.zeros(5)
    r = np.array([0.85004282, 1.26361957, 1.01875582, 1.2661551 , 0.8641883])
    K = np.array([0.64663175, 0.62005377, 0.57214239, 0.3842672 , 0.3116877])
    tf = 200
    dt = 0.01
    SPC_THRES = 1e-6
    assert np.allclose(Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES), target)

    target = targets[1]
    spc_init = np.array([0.5 for i in range(5)])
    res_init = np.array([0.1 for i in range(6)])
    b = np.array([0.9, 0.8, 1.1, 1.0, 1.0])*4
    c = np.array([[0.85860914, 1.17176702, 0.94347441, 1.10866994, 1.2371759 ,
            1.27852415],
           [1.02238836, 1.01948768, 1.25428849, 0.90014124, 0.72827434,
            0.64642639],
           [0.95163552, 0.94208151, 0.91970363, 0.86037937, 0.84526805,
            0.84374076],
           [1.03884027, 1.19774687, 1.08459477, 1.02244662, 1.30889132,
            1.0328177 ],
           [0.82889651, 0.99760798, 1.13576373, 1.02281603, 0.81106549,
            1.03599141]])
    w = np.array([1 for i in range(6)])
    m = np.array([0.48960319, 0.57932042, 0.49779724, 0.44603161, 0.63076135])
    r = np.array([1.13792535, 0.71481788, 0.92852849, 1.1630474 , 0.93047131,
           1.09396219])
    K = np.array([1.72557091, 1.86795221, 1.86117375, 1.81071033, 1.77889555,
           2.06111753])
    tf = 200
    dt = 0.01
    SPC_THRES = 1e-6
    assert np.allclose(Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES), target)

    target = targets[2]
    spc_init = np.array([0.4 for i in range(5)])
    res_init = np.array([0.2 for i in range(10)])
    b = np.array([0.9, 0.8, 1.1, 1.0, 1.0])*4
    c = np.array([[1.0602643 , 1.07868373, 0.70084849, 0.87026924, 1.23793334,
            1.27019359, 1.02624243, 1.2444197 , 1.15088166, 1.36739505],
           [0.8520995 , 0.87746294, 1.05657967, 0.85920931, 0.67309043,
            0.9012853 , 1.09495138, 0.84172396, 1.11230972, 1.21185816],
           [1.59237195, 1.00052901, 1.19167086, 1.21982551, 0.97614248,
            1.06940695, 0.91894559, 0.79603321, 1.21270515, 1.16589103],
           [0.9442301 , 0.9094415 , 1.13126104, 1.14479581, 1.29529536,
            0.90346675, 0.79667304, 1.23079451, 0.8910446 , 0.79275198],
           [1.22010188, 1.17259114, 0.8753312 , 1.12654003, 1.9044324 ,
            1.09951092, 0.69305147, 0.83562566, 1.09511894, 1.41744965]])
    w = np.array([1 for i in range(10)])
    m = np.array([0.4609046 , 0.40625631, 0.51583364, 0.47573744, 0.40025639])
    r = np.array([1.10137987, 0.74092458, 1.392985  , 0.75843837, 0.91337016,
           0.83953648, 1.12257021, 1.03624413, 1.1822436 , 1.24971757])
    K = np.array([1.89075267, 2.05734909, 1.86812723, 1.66947805, 1.9573865 ,
           2.02042697, 1.95724442, 1.61388709, 1.85837379, 2.3939331 ])
    tf = 200
    dt = 0.01
    SPC_THRES = 1e-6
    assert np.allclose(Simulate(spc_init, res_init, b, c, w, m, r, K, tf, dt, SPC_THRES), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e