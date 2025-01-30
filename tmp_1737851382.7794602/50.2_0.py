import numpy as np

# Background: The Monte Carlo method is a computational algorithm that relies on repeated random sampling to obtain numerical results. 
# In the context of finding the thermal equilibrium state of a spin system, we use the Metropolis algorithm, a type of Monte Carlo method.
# The Metropolis algorithm involves randomly selecting a spin, calculating the change in energy if the spin is flipped, and deciding 
# whether to accept the flip based on the Boltzmann distribution. The probability of accepting a spin flip is given by 
# P = exp(-ΔE / (kT)), where ΔE is the change in energy, k is the Boltzmann constant (often set to 1 in simulations), and T is the temperature.
# If ΔE is negative (the energy decreases), the flip is always accepted. If ΔE is positive, the flip is accepted with probability P.
# This process is repeated for a large number of steps to allow the system to reach thermal equilibrium.


def find_equilibrium(spins, N, T, J, num_steps):
    '''Find the thermal equilibrium state of a given spin system
    Input:
    spins: starting spin state, 1D array of 1 and -1
    N: size of spin system, int
    T: temperature, float
    J: interaction matrix, 2D array of floats
    num_steps: number of sampling steps per spin in the Monte Carlo simulation, int
    Output:
    spins: final spin state after Monte Carlo simulation, 1D array of 1 and -1
    '''
    for step in range(num_steps * N):
        # Randomly select a spin to flip
        i = np.random.randint(0, N)
        
        # Calculate the change in energy if this spin is flipped
        delta_E = 0
        for j in range(N):
            if i != j:
                delta_E += 2 * spins[i] * spins[j] * J[i, j]
        
        # Decide whether to flip the spin
        if T != 0:
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                spins[i] = -spins[i]
        else:
            if delta_E < 0:
                spins[i] = -spins[i]
    
    return spins



# Background: In statistical physics, the overlap between two spin configurations is a measure of their similarity. 
# For two spin configurations, the overlap is defined as the normalized dot product of the two spin vectors. 
# Given two spin vectors S1 and S2, each of length N, the overlap q is calculated as:
# q = (1/N) * sum(S1[i] * S2[i] for i in range(N))
# This value ranges from -1 to 1, where 1 indicates identical configurations, -1 indicates completely opposite configurations, 
# and 0 indicates no correlation. In the context of the replica method, calculating overlaps between different replicas 
# helps in understanding the distribution of states and the nature of the phase space.


def calculate_overlap(replicas):
    '''Calculate all overlaps in an ensemble of replicas
    Input:
    replicas: list of replicas, list of 1D arrays of 1 and -1
    Output:
    overlaps: pairwise overlap values between all replicas, 1D array of floats, sorted
    '''
    num_replicas = len(replicas)
    overlaps = []
    
    # Calculate pairwise overlaps
    for i in range(num_replicas):
        for j in range(i + 1, num_replicas):
            overlap = np.dot(replicas[i], replicas[j]) / len(replicas[i])
            overlaps.append(overlap)
    
    # Sort the overlaps in ascending order
    overlaps.sort()
    
    return np.array(overlaps)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('50.2', 3)
target = targets[0]

replicas = [np.ones(5) for _ in range(5)]
overlaps = calculate_overlap(replicas)
assert np.allclose(overlaps, target)
target = targets[1]

np.random.seed(1)
replicas = [np.random.choice([-1, 1], size=10) for _ in range(10)]
overlaps = calculate_overlap(replicas)
assert np.allclose(overlaps, target)
target = targets[2]

np.random.seed(3)
replicas = [np.random.choice([-1, 1], size=360) for _ in range(10)]
overlaps = calculate_overlap(replicas)
assert np.allclose(overlaps, target)
