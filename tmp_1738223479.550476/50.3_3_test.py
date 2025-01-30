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


    num_replicas = len(replicas)
    overlaps = []

    # Calculate overlaps and insert them into a sorted list using bisect
    for i in range(num_replicas):
        for j in range(i + 1, num_replicas):
            overlap = sum(a * b for a, b in zip(replicas[i], replicas[j])) / len(replicas[i])
            bisect.insort(overlaps, overlap)

    return overlaps



def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution identifies broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''



    # Normalize overlaps to have zero mean and unit variance
    overlaps_normalized = (overlaps - np.mean(overlaps)) / np.std(overlaps)

    # Reshape data for clustering
    overlaps_reshaped = overlaps_normalized.reshape(-1, 1)

    # Use DBSCAN to identify clusters in the overlap distribution
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(overlaps_reshaped)

    # Unique clusters found, ignoring noise points labeled as -1
    unique_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)

    # Potential RSB is indicated if there are multiple clusters
    potential_RSB = unique_clusters > 1

    return potential_RSB


try:
    targets = process_hdf5_to_tuple('50.3', 3)
    target = targets[0]
    N = 100
    overlaps = np.random.normal(0, 1/np.sqrt(N), size=1000)
    potential_RSB = analyze_rsb(overlaps, N)
    assert potential_RSB == target

    target = targets[1]
    N = 100
    overlaps = np.random.normal(0, 1/np.sqrt(N/3), size=1000)
    potential_RSB = analyze_rsb(overlaps, N)
    assert potential_RSB == target

    target = targets[2]
    N = 100
    samples_peak1 = np.random.normal(loc=-0.5, scale=0.2, size=500)
    samples_peak2 = np.random.normal(loc=0.5, scale=0.2, size=500)
    rsb_samples = np.concatenate([samples_peak1, samples_peak2])
    overlaps = np.clip(rsb_samples, -1, 1)
    potential_RSB = analyze_rsb(overlaps, N)
    assert potential_RSB == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e