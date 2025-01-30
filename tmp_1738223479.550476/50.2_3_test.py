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

    def calculate_energy(spins):
        '''Calculate the energy of the current spin configuration.'''
        return -0.5 * np.dot(np.dot(spins, J), spins)

    current_energy = calculate_energy(spins)

    for step in range(num_steps):
        # Randomly select a pair of spins
        i, j = np.random.choice(N, size=2, replace=False)
        
        # Flip both spins
        spins[i] *= -1
        spins[j] *= -1
        
        # Calculate the new energy
        new_energy = calculate_energy(spins)
        
        # Calculate the change in energy
        delta_E = new_energy - current_energy
        
        # Metropolis criterion: accept the change with certain probability
        if delta_E > 0 and np.random.rand() >= np.exp(-delta_E / T):
            # Revert the spin flip if not accepted
            spins[i] *= -1
            spins[j] *= -1
        else:
            # Update current energy if flip is accepted
            current_energy = new_energy

    return spins



def calculate_overlap(replicas):
    '''Calculate all overlaps in an ensemble of replicas
    Input:
    replicas: list of replicas, list of 1D arrays of 1 and -1
    Output:
    overlaps: pairwise overlap values between all replicas, 1D array of floats, sorted
    '''
    overlaps = []

    # Iterate over all pairs using the length of the replicas
    num_replicas = len(replicas)
    for i in range(num_replicas - 1):
        replica_i = replicas[i]
        len_i = len(replica_i)  # Cache the length to avoid recalculating
        for j in range(i + 1, num_replicas):
            # Direct calculation without numpy
            replica_j = replicas[j]
            dot_product = sum(x * y for x, y in zip(replica_i, replica_j))
            overlap = dot_product / len_i
            overlaps.append(overlap)

    # Use Python's built-in sorted function
    overlaps = sorted(overlaps)

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