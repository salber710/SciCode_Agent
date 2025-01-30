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




def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    # Calculate the interquartile range of the overlap distribution
    interquartile_range = iqr(overlaps)
    
    # Calculate the median of the overlap distribution
    median_value = np.median(overlaps)
    
    # Define thresholds for interquartile range and median to determine broad or multimodal distribution
    iqr_threshold = (np.max(overlaps) - np.min(overlaps)) * 0.25
    median_threshold = (np.max(overlaps) + np.min(overlaps)) / 2
    
    # Check if the interquartile range is broad and the median is not centered
    potential_RSB = interquartile_range > iqr_threshold and not (median_value - 0.1 < median_threshold < median_value + 0.1)
    
    return potential_RSB



# Background: The Sherrington-Kirkpatrick (SK) model is a type of spin glass model where each spin interacts with every other spin with a random coupling. 
# The replica method is used to study the equilibrium properties of such disordered systems by considering multiple copies (replicas) of the system. 
# In this context, the overlap between replicas is a measure of similarity, and its distribution can indicate replica symmetry breaking (RSB), 
# a phenomenon where the symmetry between replicas is broken, leading to a complex energy landscape. 
# The overlap distribution is analyzed to identify RSB by looking for broad or multimodal distributions. 
# The mean and standard deviation of the overlap distribution provide additional statistical insights into the system's behavior.



def spin_glass(N, T, num_steps, num_replicas, num_realizations):
    '''Simulation the SK model using replica method, analyze overlap and identify potential replica symmetry breaking
    Input:
    N: size of spin system, int
    T: temperature, float
    num_steps: number of sampling steps per spin in the Monte Carlo simulation, int
    num_replicas: number of system replicas in one realization, int
    num_realizations: number of realizations to sample different J's, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    mean: mean of the overall overlap distribution, float
    std: standard deviation of the overall overlap distribution, float
    '''
    
    beta = 1 / T
    all_overlaps = []

    for _ in range(num_realizations):
        # Generate random interaction coefficients J_ij
        J = np.random.randn(N, N)
        J = (J + J.T) / 2  # Make J symmetric
        np.fill_diagonal(J, 0)  # No self-interaction

        replicas = []

        for _ in range(num_replicas):
            # Initialize spins randomly
            spins = np.random.choice([-1, 1], size=N)
            # Find equilibrium state using Monte Carlo method
            for _ in range(num_steps):
                for _ in range(N):
                    i = np.random.randint(0, N)
                    delta_E = 2 * spins[i] * np.dot(J[i], spins)
                    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                        spins[i] = -spins[i]
            replicas.append(spins)

        # Calculate overlaps for this realization
        overlaps = np.array([np.inner(replicas[i], replicas[j]) / np.sqrt(np.inner(replicas[i], replicas[i]) * np.inner(replicas[j], replicas[j]))
                             for i in range(num_replicas) for j in range(i + 1, num_replicas)])
        all_overlaps.extend(overlaps)

    # Analyze the overall overlap distribution
    all_overlaps = np.array(all_overlaps)
    mean = np.mean(all_overlaps)
    std = np.std(all_overlaps)
    
    # Determine potential RSB
    potential_RSB = analyze_rsb(all_overlaps, N)

    return potential_RSB, mean, std


try:
    targets = process_hdf5_to_tuple('50.4', 3)
    target = targets[0]
    np.random.seed(1)
    T = 1.5
    N = 100
    num_steps = 500
    num_replicas = 50
    num_realizations = 10
    aa, bb, cc = spin_glass(N, T, num_steps, num_replicas, num_realizations)
    a, b, c = target
    assert a == aa and np.allclose((b, c), (bb, cc))

    target = targets[1]
    np.random.seed(3)
    T = 0.7
    N = 100
    num_steps = 500
    num_replicas = 50
    num_realizations = 10
    aa, bb, cc = spin_glass(N, T, num_steps, num_replicas, num_realizations)
    a, b, c = target
    assert a == aa and np.allclose((b, c), (bb, cc))

    target = targets[2]
    np.random.seed(2)
    T = 0.5
    N = 256
    num_steps = 500
    num_replicas = 50
    num_realizations = 5
    aa, bb, cc = spin_glass(N, T, num_steps, num_replicas, num_realizations)
    a, b, c = target
    assert a == aa and np.allclose((b, c), (bb, cc))

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e