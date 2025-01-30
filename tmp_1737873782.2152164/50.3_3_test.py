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
    '''Analyze the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    # Calculate histogram of the overlap distribution
    histogram, bin_edges = np.histogram(overlaps, bins='auto', density=True)

    # Identify peaks in the histogram by finding local maxima
    peaks = []
    for i in range(1, len(histogram) - 1):
        if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
            peaks.append(histogram[i])

    # Determine potential RSB by checking for multiple significant peaks
    # A rough criteria for RSB is having more than one significant peak
    # Significance can be determined by thresholding the peak height
    threshold = 0.1 * max(histogram) # Example threshold as 10% of max height
    significant_peaks = [peak for peak in peaks if peak > threshold]

    # If there are multiple significant peaks, RSB is likely
    potential_RSB = len(significant_peaks) > 1

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