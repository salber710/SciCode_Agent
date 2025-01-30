from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def find_equilibrium(spins, N, T, J, num_steps):
    beta = 1 / T

    for _ in range(num_steps):
        for _ in range(N):
            i = np.random.randint(0, N)
            # Calculate the energy change for flipping spin i
            delta_E = 2 * spins[i] * np.dot(J[i], spins)
            # Accept or reject the flip based on a modified criterion
            if delta_E < 0:
                spins[i] = -spins[i]
            elif np.random.rand() < (1 - np.exp(-beta * delta_E)):
                spins[i] = -spins[i]
    return spins



# Background: In statistical physics, the concept of overlap is used to measure the similarity between different configurations or states of a system. 
# For spin systems, the overlap between two replicas (configurations) is defined as the normalized dot product of their spin vectors. 
# This measure ranges from -1 to 1, where 1 indicates identical configurations, -1 indicates completely opposite configurations, 
# and 0 indicates no correlation. In the context of the replica method, calculating overlaps between replicas helps in understanding 
# the distribution of states and the nature of the phase space, especially in disordered systems like spin glasses.


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
            # Calculate the overlap as the normalized dot product
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