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

    # Boltzmann constant
    k_B = 1.0

    # Monte Carlo simulation
    for step in range(num_steps * N):
        # Randomly choose a spin to flip
        i = np.random.randint(N)

        # Calculate the change in energy if the spin is flipped
        delta_E = 2 * spins[i] * sum(J[i, j] * spins[j] for j in range(N))

        # Decide whether to flip the spin
        if delta_E < 0:
            # Flip the spin since the energy is lowered
            spins[i] *= -1
        else:
            # Flip the spin with probability exp(-delta_E / (k_B * T))
            if np.random.rand() < np.exp(-delta_E / (k_B * T)):
                spins[i] *= -1

    return spins




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

    return overlaps


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e