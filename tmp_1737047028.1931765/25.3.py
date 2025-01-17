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



# Background: In ecological modeling, simulating the dynamics of species and resources over time involves integrating their respective growth and consumption rates. 
# The MacArthur consumer-resource model provides a framework for understanding how species grow based on resource availability, while resources themselves follow logistic growth dynamics.
# The simulation involves iteratively updating species and resource levels over discrete time steps from an initial state to a final time.
# The species growth is determined by the balance between resource consumption and maintenance costs, while resource levels change due to logistic growth and consumption by species.
# A species is considered extinct if its abundance falls below a specified threshold (SPC_THRES) during the simulation.


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
    t = 0.0
    while t < tf:
        # Calculate species growth rates
        effective_resource_contribution = np.dot(c, w * res)
        g_spc = b * (effective_resource_contribution - m)
        
        # Update species levels
        spc += g_spc * spc * dt
        
        # Calculate resource growth rates
        logistic_growth = r * res * (1 - res / K)
        consumption = np.dot(spc, c)
        f_res = logistic_growth - consumption
        
        # Update resource levels
        res += f_res * dt
        
        # Increment time
        t += dt
    
    # Determine surviving species
    survivors = [i for i in range(len(spc)) if spc[i] > SPC_THRES]
    
    return survivors


from scicode.parse.parse import process_hdf5_to_tuple

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
