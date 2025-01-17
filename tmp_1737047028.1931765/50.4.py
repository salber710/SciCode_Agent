import numpy as np

# Background: The Monte Carlo method is a computational algorithm that relies on repeated random sampling to obtain numerical results. 
# In the context of finding the thermal equilibrium state of a spin system, we use the Metropolis algorithm, a type of Monte Carlo method.
# The system is represented by spins that can be either +1 or -1. The energy of the system is determined by the interaction coefficients J_ij.
# At each step, a spin is randomly selected and flipped, and the change in energy (ΔE) is calculated. 
# If ΔE is negative, the flip is accepted because it lowers the system's energy. 
# If ΔE is positive, the flip is accepted with a probability of exp(-ΔE / (k_B * T)), where k_B is the Boltzmann constant and T is the temperature.
# This process is repeated for a number of steps to allow the system to reach thermal equilibrium.


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
    
    k_B = 1.0  # Boltzmann constant, can be set to 1 for simplicity in this context

    for step in range(num_steps * N):
        # Randomly select a spin to flip
        i = np.random.randint(0, N)
        
        # Calculate the change in energy if this spin is flipped
        delta_E = 0
        for j in range(N):
            if i != j:
                delta_E += 2 * spins[i] * spins[j] * J[i, j]
        
        # Decide whether to accept the flip
        if delta_E < 0:
            # Accept the flip
            spins[i] *= -1
        else:
            # Accept the flip with a probability of exp(-delta_E / (k_B * T))
            if np.random.rand() < np.exp(-delta_E / (k_B * T)):
                spins[i] *= -1

    return spins


# Background: In statistical physics, the overlap between two spin configurations is a measure of their similarity.
# It is defined as the normalized dot product of the two spin vectors. For a system of N spins, the overlap q between
# two configurations s^a and s^b is given by q = (1/N) * sum(s^a_i * s^b_i) for i from 1 to N.
# In the context of the replica method, calculating overlaps between replicas helps in understanding the structure
# of the phase space and the nature of the equilibrium states. The overlaps are useful in studying phenomena like
# spin glass behavior, where multiple equilibrium states can exist.

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

    # Sort the overlaps in ascending order
    overlaps.sort()

    return overlaps


# Background: In the context of spin systems and the replica method, Replica Symmetry Breaking (RSB) is a concept that arises in the study of disordered systems, such as spin glasses. RSB occurs when the overlap distribution of spin configurations (replicas) exhibits certain characteristics, such as a broad peak or multimodal behavior, indicating that the system does not settle into a single equilibrium state but rather into multiple states. This can be identified by analyzing the distribution of overlaps between replicas. A broad or multimodal distribution suggests the presence of multiple equilibrium states, which is a hallmark of RSB.

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

    # Analyze the histogram to determine if there is a broad peak or multimodal behavior
    # A simple approach is to check for multiple peaks in the histogram
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks += 1

    # If there are multiple peaks, it suggests potential RSB
    potential_RSB = peaks > 1

    return potential_RSB



# Background: The Sherrington-Kirkpatrick (SK) model is a type of spin glass model where each pair of spins has a random interaction coefficient.
# In this model, the interaction coefficients J_ij are drawn from a normal distribution with mean 0 and variance 1/N.
# The SK model is used to study disordered systems and phenomena like replica symmetry breaking (RSB).
# In this simulation, we will generate multiple realizations of the SK model, each with a different set of interaction coefficients.
# For each realization, we will use the replica method to sample the equilibrium state distribution and calculate overlaps between replicas.
# By analyzing the overlap distribution across all realizations, we can identify potential RSB and calculate the mean and standard deviation of the overlaps.

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
        # Generate a random interaction matrix J for this realization
        J = np.random.randn(N, N) / np.sqrt(N)
        J = (J + J.T) / 2  # Make J symmetric

        # Create replicas and find their equilibrium states
        replicas = []
        for _ in range(num_replicas):
            # Initialize spins randomly
            spins = np.random.choice([-1, 1], size=N)
            # Find equilibrium state using the Monte Carlo method
            spins = find_equilibrium(spins, N, T, J, num_steps)
            replicas.append(spins)

        # Calculate overlaps for this realization
        overlaps = calculate_overlap(replicas)
        all_overlaps.extend(overlaps)

    # Analyze the overall overlap distribution
    potential_RSB = analyze_rsb(all_overlaps, N)

    # Calculate mean and standard deviation of the overall overlap distribution
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
