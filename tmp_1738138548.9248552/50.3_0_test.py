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



def calculate_overlap(replicas):
    overlaps = np.array([np.inner(replicas[i], replicas[j]) / np.sqrt(np.inner(replicas[i], replicas[i]) * np.inner(replicas[j], replicas[j]))
                         for i in range(len(replicas)) for j in range(i + 1, len(replicas))])
    return np.sort(overlaps)



# Background: Replica symmetry breaking (RSB) is a concept from statistical physics, particularly in the study of spin glasses and disordered systems. 
# It refers to a situation where the symmetry between different replicas (copies) of a system is broken, leading to a complex energy landscape with many local minima.
# In the context of spin glasses, RSB can be detected by analyzing the distribution of overlaps between different replicas of the system.
# A broad peak or multimodal distribution in the overlap values can indicate RSB, as it suggests the presence of multiple distinct states or configurations that the system can occupy.
# The overlap is a measure of similarity between two configurations, and a wide or multimodal distribution suggests that the system does not settle into a single configuration.

def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    # Calculate the histogram of overlaps
    hist, bin_edges = np.histogram(overlaps, bins='auto', density=True)
    
    # Check for broad peak or multimodal distribution
    # A simple heuristic: if there are multiple peaks in the histogram, it might indicate RSB
    # We can count the number of local maxima in the histogram
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks += 1
    
    # If there are more than one peak, it suggests a multimodal distribution
    potential_RSB = peaks > 1
    
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