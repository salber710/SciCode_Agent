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

    # Fit a Kernel Density Estimator to the normalized overlaps
    kde = KernelDensity(kernel='tophat', bandwidth=0.1).fit(overlaps_normalized[:, None])

    # Evaluate the density on a grid and calculate the log density
    grid = np.linspace(min(overlaps_normalized), max(overlaps_normalized), 1000)[:, None]
    log_density = kde.score_samples(grid)
    density = np.exp(log_density)

    # Find peaks in the density
    peaks = (np.diff(np.sign(np.diff(density))) < 0).nonzero()[0] + 1

    # Consider peaks that are prominent enough
    prominence_threshold = 0.05 * np.max(density)
    significant_peaks = [p for p in peaks if density[p] > prominence_threshold]

    # Potential RSB is indicated if there is more than one significant peak
    potential_RSB = len(significant_peaks) > 1

    return potential_RSB



# Background: The Sherrington-Kirkpatrick (SK) model is a type of spin glass system where spins can interact with each other via randomly distributed coupling constants, J_ij. At a given temperature T, the system can exhibit complex behavior like replica symmetry breaking (RSB), where the overlap distribution of spin configurations sampled from different realizations of J's can become multimodal. The overlap between two replicas is a measure of similarity in their spin configurations. Analyzing the overlap distribution helps to identify RSB, which is indicated by a broad peak or multiple peaks in the distribution. The simulation involves generating multiple replicas and realizations of the system, calculating overlaps, and analyzing their statistical properties.



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
    
    # Function to find equilibrium state of a spin system
    def find_equilibrium(spins, N, T, J, num_steps):
        def calculate_energy(spins):
            return -0.5 * np.dot(np.dot(spins, J), spins)
        
        current_energy = calculate_energy(spins)
        
        for step in range(num_steps):
            i, j = np.random.choice(N, size=2, replace=False)
            spins[i] *= -1
            spins[j] *= -1
            new_energy = calculate_energy(spins)
            delta_E = new_energy - current_energy
            if delta_E > 0 and np.random.rand() >= np.exp(-delta_E / T):
                spins[i] *= -1
                spins[j] *= -1
            else:
                current_energy = new_energy
                
        return spins
    
    # Function to calculate overlaps
    def calculate_overlap(replicas):
        num_replicas = len(replicas)
        overlaps = []
        for i in range(num_replicas):
            for j in range(i + 1, num_replicas):
                overlap = sum(a * b for a, b in zip(replicas[i], replicas[j])) / len(replicas[i])
                overlaps.append(overlap)
        return overlaps
    
    all_overlaps = []
    
    for realization in range(num_realizations):
        # Define interaction matrix J with Gaussian random entries
        J = np.random.randn(N, N)
        J = (J + J.T) / 2  # Make it symmetric
        np.fill_diagonal(J, 0)  # No self-interaction
        
        replicas = []
        
        for replica in range(num_replicas):
            spins = np.random.choice([-1, 1], size=N)  # Random initial state
            spins = find_equilibrium(spins, N, T, J, num_steps)
            replicas.append(spins)
        
        overlaps = calculate_overlap(replicas)
        all_overlaps.extend(overlaps)
    
    # Analyze replica symmetry breaking
    overlaps_array = np.array(all_overlaps)
    mean = np.mean(overlaps_array)
    std = np.std(overlaps_array)
    
    # Normalize overlaps
    overlaps_normalized = (overlaps_array - mean) / std
    
    # KDE to find peaks
    kde = KernelDensity(kernel='tophat', bandwidth=0.1).fit(overlaps_normalized[:, None])
    grid = np.linspace(min(overlaps_normalized), max(overlaps_normalized), 1000)[:, None]
    log_density = kde.score_samples(grid)
    density = np.exp(log_density)
    
    # Find peaks
    peaks = (np.diff(np.sign(np.diff(density))) < 0).nonzero()[0] + 1
    prominence_threshold = 0.05 * np.max(density)
    significant_peaks = [p for p in peaks if density[p] > prominence_threshold]
    
    potential_RSB = len(significant_peaks) > 1
    
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