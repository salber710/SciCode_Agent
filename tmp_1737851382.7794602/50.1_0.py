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
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            spins[i] = -spins[i]
    
    return spins

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('50.1', 3)
target = targets[0]

np.random.seed(1)
N = 50
T = 1.5
num_steps = 1000
spins = np.random.choice([-1, 1], size=N)
J = np.random.randn(N, N)
J = (J + J.T) / 2  # Make symmetric
np.fill_diagonal(J, 0)  # No self-interaction
J = J / np.sqrt(N)
spins = find_equilibrium(spins, N, T, J, num_steps)
assert np.allclose(spins, target)
target = targets[1]

np.random.seed(2)
N = 50
T = 0.5
num_steps = 1000
spins = np.random.choice([-1, 1], size=N)
J = np.random.randn(N, N)
J = (J + J.T) / 2  # Make symmetric
np.fill_diagonal(J, 0)  # No self-interaction
J = J / np.sqrt(N)
spins = find_equilibrium(spins, N, T, J, num_steps)
assert np.allclose(spins, target)
target = targets[2]

np.random.seed(2)
N = 100
T = 0.7
num_steps = 5000
spins = np.random.choice([-1, 1], size=N)
J = np.random.randn(N, N)
J = (J + J.T) / 2  # Make symmetric
np.fill_diagonal(J, 0)  # No self-interaction
J = J / np.sqrt(N)
spins = find_equilibrium(spins, N, T, J, num_steps)
assert np.allclose(spins, target)
