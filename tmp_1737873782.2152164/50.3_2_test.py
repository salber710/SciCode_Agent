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

    # Sort overlaps
    overlaps.sort()

    return overlaps




def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution indicates potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realizations, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    # Calculate the mean and standard deviation of the overlaps
    mean_overlap = np.mean(overlaps)
    std_overlap = np.std(overlaps)
    
    # Analyze the distribution of overlaps
    # A broad peak or multimodal behavior can be indicated by a high standard deviation
    # relative to the mean, or the presence of distinct clusters in the data

    # Set a threshold for potential RSB based on the standard deviation
    # This threshold is somewhat arbitrary and may need tuning based on the specific system
    threshold = 0.1  # Example threshold, can be adjusted based on further analysis
    
    # Check if the standard deviation exceeds the threshold
    potential_RSB = std_overlap / mean_overlap > threshold

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