from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The Monte Carlo method is a computational algorithm that relies on repeated random sampling to obtain numerical results. 
# In the context of finding the thermal equilibrium state of a spin system, the Metropolis algorithm is often used. 
# This algorithm involves flipping spins randomly and deciding whether to accept the flip based on the change in energy and the temperature of the system.
# The probability of accepting a spin flip that increases the system's energy is given by the Boltzmann factor, exp(-ΔE / (kT)), where ΔE is the change in energy, 
# k is the Boltzmann constant (often set to 1 in simulations), and T is the temperature. 
# The system is considered to be in thermal equilibrium when the macroscopic properties do not change with further simulation steps.


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
    
    # Boltzmann constant is set to 1 for simplicity
    k = 1.0
    
    for step in range(num_steps * N):
        # Randomly select a spin to flip
        i = np.random.randint(0, N)
        
        # Calculate the change in energy if this spin is flipped
        delta_E = 2 * spins[i] * np.sum(J[i] * spins)
        
        # Decide whether to flip the spin
        if delta_E < 0:
            # If the energy decreases, always accept the flip
            spins[i] *= -1
        else:
            # If the energy increases, accept the flip with a probability exp(-ΔE / (kT))
            if np.random.rand() < np.exp(-delta_E / (k * T)):
                spins[i] *= -1
    
    return spins


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e