import numpy as np

# Background: The Monte Carlo method is a computational algorithm that relies on repeated random sampling to obtain numerical results. 
# In the context of finding the thermal equilibrium state of a spin system, we use the Metropolis algorithm, a type of Monte Carlo method.
# The Metropolis algorithm involves randomly selecting a spin, calculating the change in energy if the spin is flipped, and deciding 
# whether to accept the flip based on the Boltzmann distribution. The probability of accepting a spin flip is given by 
# P = exp(-ΔE / (kT)), where ΔE is the change in energy, k is the Boltzmann constant (often set to 1 in simulations), and T is the temperature.
# If ΔE is negative (the energy decreases), the flip is always accepted. If ΔE is positive, the flip is accepted with probability P.
# This process is repeated for a large number of steps to allow the system to reach thermal equilibrium.


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
    for step in range(num_steps * N):
        # Randomly select a spin to flip
        i = np.random.randint(0, N)
        
        # Calculate the change in energy if this spin is flipped
        delta_E = 0
        for j in range(N):
            if i != j:
                delta_E += 2 * spins[i] * spins[j] * J[i, j]
        
        # Decide whether to flip the spin
        if T != 0:
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                spins[i] = -spins[i]
        else:
            if delta_E < 0:
                spins[i] = -spins[i]
    
    return spins


# Background: In statistical physics, the overlap between two spin configurations is a measure of their similarity. 
# For two spin configurations, the overlap is defined as the normalized dot product of the two spin vectors. 
# Given two spin vectors S1 and S2, each of length N, the overlap q is calculated as:
# q = (1/N) * sum(S1[i] * S2[i] for i in range(N))
# This value ranges from -1 to 1, where 1 indicates identical configurations, -1 indicates completely opposite configurations, 
# and 0 indicates no correlation. In the context of the replica method, calculating overlaps between different replicas 
# helps in understanding the distribution of states and the nature of the phase space.


def calculate_overlap(replicas):
    '''Calculate all overlaps in an ensemble of replicas
    Input:
    replicas: list of replicas, list of 1D arrays of 1 and -1
    Output:
    overlaps: pairwise overlap values between all replicas, 1D array of floats, sorted
    '''
    if not replicas:
        return np.array([])

    # Check for consistent lengths
    length = len(replicas[0])
    if any(len(replica) != length for replica in replicas):
        raise ValueError("All replicas must have the same length")

    num_replicas = len(replicas)
    overlaps = []
    
    # Calculate pairwise overlaps
    for i in range(num_replicas):
        for j in range(i + 1, num_replicas):
            overlap = np.dot(replicas[i], replicas[j]) / length
            overlaps.append(overlap)
    
    # Sort the overlaps in ascending order
    overlaps.sort()
    
    return np.array(overlaps)


# Background: In the context of spin systems and statistical physics, Replica Symmetry Breaking (RSB) is a concept that arises in the study of disordered systems, such as spin glasses. RSB refers to a situation where the symmetry between different replicas of a system is broken, leading to a complex energy landscape with many local minima. This is often indicated by a broad or multimodal distribution of overlaps between replicas. A broad peak in the overlap distribution suggests a wide variety of states with similar energies, while multimodal behavior indicates distinct groups of states, both of which are signatures of RSB. Analyzing the overlap distribution can help identify these characteristics.

def analyze_rsb(overlaps, N):
    '''Analyze if the overlap distribution to identify broad peak or multimodal behavior, indicating potential replica symmetry breaking.
    Input:
    overlaps: all overlap values between replicas from all realization, 1D array of floats
    N: size of spin system, int
    Output:
    potential_RSB: True if potential RSB is indicated, False otherwise, boolean
    '''


    if len(overlaps) == 0:
        return False

    # Calculate the histogram of overlaps
    hist, bin_edges = np.histogram(overlaps, bins='auto', density=True)

    # Analyze the histogram for broad peaks or multimodal behavior
    # A simple approach is to check for multiple peaks in the histogram
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks += 1

    # If there are multiple peaks, it suggests potential RSB
    potential_RSB = peaks > 1

    # Check for broad peaks by analyzing the width of the peaks
    # A broad peak is considered if the width is greater than 25% of the range of overlaps
    peak_widths = []
    in_peak = False
    start = 0
    for i in range(1, len(hist)):
        if hist[i] > hist[i - 1] and not in_peak:
            start = i
            in_peak = True
        elif hist[i] < hist[i - 1] and in_peak:
            peak_widths.append(bin_edges[i] - bin_edges[start])
            in_peak = False

    if in_peak:
        peak_widths.append(bin_edges[-1] - bin_edges[start])

    broad_peak = any(width > 0.25 * (overlaps.max() - overlaps.min()) for width in peak_widths)

    return potential_RSB or broad_peak



# Background: The Sherrington-Kirkpatrick (SK) model is a type of spin glass model where each pair of spins has a random interaction coefficient. 
# The SK model is used to study disordered magnetic systems. In this model, the interaction coefficients J_ij are drawn from a normal distribution. 
# The replica method involves creating multiple copies (replicas) of the system to study the statistical properties of the spin configurations. 
# By simulating the SK model at a given temperature T, we can use the Monte Carlo method to find the equilibrium state of each replica. 
# The overlap between replicas is calculated to analyze the distribution of states. A broad or multimodal overlap distribution can indicate 
# replica symmetry breaking (RSB), which is a hallmark of complex energy landscapes in disordered systems. The mean and standard deviation 
# of the overlap distribution provide insights into the nature of the spin configurations.


def spin_glass(N, T, num_steps, num_replicas, num_realizations):
    '''Simulation the SK model using replica method, analyze overlap and identify potential replica symmetry breaking
    Input:
    N: size of spin system, int
    T: temprature, float
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
        # Generate random interaction matrix J with elements drawn from a normal distribution
        J = np.random.randn(N, N)
        J = (J + J.T) / 2  # Make J symmetric
        np.fill_diagonal(J, 0)  # No self-interaction

        replicas = []
        for _ in range(num_replicas):
            # Initialize spins randomly
            spins = np.random.choice([-1, 1], size=N)
            # Find equilibrium state using the Metropolis algorithm
            spins = find_equilibrium(spins, N, T, J, num_steps)
            replicas.append(spins)

        # Calculate overlaps for this realization
        overlaps = calculate_overlap(replicas)
        all_overlaps.extend(overlaps)

    all_overlaps = np.array(all_overlaps)

    # Analyze for potential RSB
    potential_RSB = analyze_rsb(all_overlaps, N)

    # Calculate mean and standard deviation of the overlap distribution
    mean = np.mean(all_overlaps)
    std = np.std(all_overlaps)

    return potential_RSB, mean, std

from scicode.parse.parse import process_hdf5_to_tuple
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
