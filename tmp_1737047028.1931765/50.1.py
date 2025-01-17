import numpy as np



# Background: The Monte Carlo method is a computational algorithm that relies on repeated random sampling to obtain numerical results. 
# In the context of finding the thermal equilibrium state of a spin system, we use the Metropolis algorithm, a type of Monte Carlo method.
# The system is represented by spins that can be either +1 or -1. The energy of the system is determined by the interaction coefficients J_ij.
# At each step, a spin is randomly selected and flipped, and the change in energy (ΔE) is calculated. 
# If ΔE is negative, the flip is accepted because it lowers the system's energy. 
# If ΔE is positive, the flip is accepted with a probability of exp(-ΔE / (k_B * T)), where k_B is the Boltzmann constant and T is the temperature.
# This process is repeated for a number of steps to allow the system to reach thermal equilibrium.


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
    
    k_B = 1.0  # Boltzmann constant, can be set to 1 for simplicity in this context

    for step in range(num_steps * N):
        # Randomly select a spin to flip
        i = np.random.randint(0, N)
        
        # Calculate the change in energy if this spin is flipped
        delta_E = 0
        for j in range(N):
            if i != j:
                delta_E += 2 * spins[i] * spins[j] * J[i, j]
        
        # Decide whether to accept the flip
        if delta_E < 0:
            # Accept the flip
            spins[i] *= -1
        else:
            # Accept the flip with a probability of exp(-delta_E / (k_B * T))
            if np.random.rand() < np.exp(-delta_E / (k_B * T)):
                spins[i] *= -1

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
