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
    '''Analyze if the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''
    # Calculate histogram of overlaps
    hist, bin_edges = np.histogram(overlaps, bins='auto', density=True)

    # Identify peaks in the histogram
    peak_indices = np.argwhere((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:])) + 1

    # Multimodal if there is more than one peak
    if len(peak_indices) > 1:
        potential_RSB = True
    else:
        potential_RSB = False

    return potential_RSB




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
    
    all_overlaps = []

    for _ in range(num_realizations):
        # Generate random interaction matrix J for each realization
        J = np.random.randn(N, N)
        J = (J + J.T) / 2  # Symmetrize the matrix
        np.fill_diagonal(J, 0)  # No self-interaction

        replicas = []

        # Generate replicas and let them reach equilibrium
        for _ in range(num_replicas):
            # Random initial spin configuration
            spins = np.random.choice([-1, 1], size=N)
            
            # Find equilibrium state using Monte Carlo simulation
            spins = find_equilibrium(spins, N, T, J, num_steps)

            replicas.append(spins)

        # Calculate overlaps for these replicas
        overlaps = calculate_overlap(replicas)
        all_overlaps.extend(overlaps)

    # Analyze if replica symmetry breaking occurs
    potential_RSB = analyze_rsb(all_overlaps, N)

    # Calculate mean and standard deviation of the overall overlap distribution
    mean = np.mean(all_overlaps)
    std = np.std(all_overlaps)

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