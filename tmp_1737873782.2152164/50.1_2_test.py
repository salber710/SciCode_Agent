from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



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
    
    # Boltzmann constant (can be set to 1 in natural units)
    k_B = 1.0

    # Monte Carlo simulation to find thermal equilibrium
    for step in range(num_steps * N):
        # Randomly select a spin to try to flip
        i = np.random.randint(N)
        
        # Calculate the change in energy if this spin is flipped
        delta_E = 2 * spins[i] * sum(J[i, j] * spins[j] for j in range(N))
        
        # Metropolis criterion
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / (k_B * T)):
            # Flip the spin
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