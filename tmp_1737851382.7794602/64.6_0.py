import numpy as np
import itertools

# Background: In computational simulations, especially those involving particles in a confined space, 
# it is common to use periodic boundary conditions (PBCs) to simulate an infinite system using a finite 
# simulation box. This is done by wrapping the coordinates of particles such that when a particle moves 
# out of one side of the box, it re-enters from the opposite side. This helps in avoiding edge effects 
# and mimics a larger, continuous space. The wrapping is typically done by taking the modulus of the 
# particle's position with respect to the box size. For a cubic box of size L, the wrapped coordinate 
# can be calculated as r' = r % L, where r is the original coordinate.


def wrap(r, L):
    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.
    Parameters:
    r : The (x, y, z) coordinates of a particle.
    L (float): The length of each side of the cubic box.
    Returns:
    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.
    '''
    if L <= 0:
        raise ValueError("Box size L must be positive.")
    if not isinstance(r, np.ndarray):
        raise TypeError("Input r must be a numpy array.")
    if r.ndim != 1 or r.size != 3:
        raise ValueError("Input r must be a 1D array with three elements.")
    if not np.issubdtype(r.dtype, np.number):
        raise TypeError("Elements of r must be numeric.")
    # Use numpy's modulus operation to wrap the coordinates
    coord = np.mod(r, L)
    return coord


# Background: In computational simulations of particles within a periodic cubic system, the concept of 
# minimum image distance is crucial for calculating interactions between particles. The minimum image 
# distance is the shortest distance between two particles considering the periodic boundary conditions. 
# In a cubic box of size L, each particle can be thought of as having infinite periodic images in all 
# directions. The minimum image convention ensures that we consider the closest image of a particle 
# when calculating distances. This is done by adjusting the distance between two particles such that 
# it is within half the box length in each dimension. Mathematically, for each dimension, the distance 
# is adjusted using the formula: d = r2 - r1, and then d = d - L * round(d / L), where round is the 
# nearest integer function. This ensures that the distance is the shortest possible considering the 
# periodic boundaries.


def dist(r1, r2, L):
    '''Calculate the minimum image distance between two atoms in a periodic cubic system.
    Parameters:
    r1 : The (x, y, z) coordinates of the first atom.
    r2 : The (x, y, z) coordinates of the second atom.
    L (float): The length of the side of the cubic box.
    Returns:
    float: The minimum image distance between the two atoms.
    '''
    if L <= 0:
        raise ValueError("Box size L must be positive.")
    if not isinstance(r1, np.ndarray) or not isinstance(r2, np.ndarray):
        raise TypeError("Inputs r1 and r2 must be numpy arrays.")
    if r1.ndim != 1 or r1.size != 3 or r2.ndim != 1 or r2.size != 3:
        raise ValueError("Inputs r1 and r2 must be 1D arrays with three elements.")
    if not np.issubdtype(r1.dtype, np.number) or not np.issubdtype(r2.dtype, np.number):
        raise TypeError("Elements of r1 and r2 must be numeric.")
    
    # Calculate the vector difference
    delta = r2 - r1
    
    # Apply the minimum image convention
    delta = delta - L * np.round(delta / L)
    
    # Ensure the distance is calculated correctly when close to box boundaries
    delta = np.where(np.abs(delta) > L / 2, delta - np.sign(delta) * L, delta)
    
    # Calculate the Euclidean distance
    distance = np.sqrt(np.sum(delta**2))
    
    return distance


# Background: The Lennard-Jones potential is a mathematical model that describes the interaction between a pair of neutral atoms or molecules. 
# It is widely used in molecular dynamics simulations to model the forces between particles. The potential is characterized by two parameters: 
# sigma (σ), which is the distance at which the potential reaches its minimum, and epsilon (ε), which is the depth of the potential well. 
# The Lennard-Jones potential is given by the formula: 
# E(r) = 4 * ε * [(σ/r)^12 - (σ/r)^6], where r is the distance between the particles. 
# The term (σ/r)^12 represents the repulsive forces due to overlapping electron orbitals, while (σ/r)^6 represents the attractive van der Waals forces. 
# The potential approaches zero as the distance r becomes much larger than σ, indicating negligible interaction at large separations.

def E_ij(r, sigma, epsilon):
    '''Calculate the Lennard-Jones potential energy between two particles.
    Parameters:
    r : float
        The distance between the two particles.
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The potential energy between the two particles at distance r.
    '''
    if r <= 0:
        raise ValueError("Distance r must be positive and non-zero.")
    if sigma <= 0:
        raise ValueError("Sigma must be positive and non-zero.")
    if epsilon < 0:
        raise ValueError("Epsilon must be non-negative.")

    # Calculate the ratio of sigma to r
    sr_ratio = sigma / r
    
    # Calculate the Lennard-Jones potential using the formula
    E_lj = 4 * epsilon * (sr_ratio**12 - sr_ratio**6)
    
    return E_lj


# Background: In molecular dynamics simulations, calculating the total energy of a single particle 
# involves summing up the interaction energies between that particle and all other particles in the system. 
# The Lennard-Jones potential, which models the interaction between a pair of particles, is used for this purpose. 
# Given the periodic boundary conditions, the minimum image distance is used to ensure that the shortest 
# possible distance is considered for each pairwise interaction. The total energy of a particle is the sum 
# of the Lennard-Jones potential energies with all other particles, considering these minimum image distances.


def E_i(r, positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a particle with other particles in a periodic system.
    Parameters:
    r : array_like
        The (x, y, z) coordinates of the target particle.
    positions : array_like
        An array of (x, y, z) coordinates for each of the other particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the particle due to its interactions with other particles.
    '''
    def dist(r1, r2, L):
        '''Calculate the minimum image distance between two atoms in a periodic cubic system.'''
        delta = r2 - r1
        delta = delta - L * np.round(delta / L)
        return np.sqrt(np.sum(delta**2))

    def E_ij(r, sigma, epsilon):
        '''Calculate the Lennard-Jones potential energy between two particles.'''
        if r == 0:
            return 0  # Avoid division by zero and self-interaction energy calculation
        sr_ratio = sigma / r
        return 4 * epsilon * (sr_ratio**12 - sr_ratio**6)

    total_energy = 0.0
    for pos in positions:
        if not np.array_equal(r, pos):  # Ensure we do not calculate self-interaction
            distance = dist(r, pos, L)
            energy = E_ij(distance, sigma, epsilon)
            total_energy += energy

    return total_energy


# Background: In molecular dynamics simulations, the total energy of a system is calculated by summing 
# the interaction energies between all pairs of particles. The Lennard-Jones potential is used to model 
# these interactions, and periodic boundary conditions are applied to simulate an infinite system. 
# The total energy of the system is the sum of the Lennard-Jones potential energies for each unique 
# pair of particles, considering the minimum image distance to account for periodic boundaries. 
# Care must be taken to avoid double-counting interactions, as each pair should only be considered once.



def E_system(positions, L, sigma, epsilon):
    '''Calculate the total Lennard-Jones potential energy of a system of particles in a periodic cubic box.
    Parameters:
    positions : array_like
        An array of (x, y, z) coordinates for each of the particles in the system.
    L : float
        The length of the side of the cubic box
    sigma : float
        The distance at which the potential minimum occurs
    epsilon : float
        The depth of the potential well
    Returns:
    float
        The total Lennard-Jones potential energy of the system.
    '''
    def dist(r1, r2, L):
        '''Calculate the minimum image distance between two atoms in a periodic cubic system.'''
        delta = r2 - r1
        if L != 0:
            delta = delta - L * np.round(delta / L)
        return np.sqrt(np.sum(delta**2))

    def E_ij(r, sigma, epsilon):
        '''Calculate the Lennard-Jones potential energy between two particles.'''
        if r == 0:
            return 0  # Avoid division by zero and self-interaction energy calculation
        sr_ratio = sigma / r
        return 4 * epsilon * (sr_ratio**12 - sr_ratio**6)

    total_E = 0.0
    # Iterate over all unique pairs of particles
    for i, j in itertools.combinations(range(len(positions)), 2):
        r1 = positions[i]
        r2 = positions[j]
        distance = dist(r1, r2, L)
        energy = E_ij(distance, sigma, epsilon)
        total_E += energy

    return total_E



# Background: Grand Canonical Monte Carlo (GCMC) simulations are used to model systems where the number of particles, volume, and temperature are allowed to fluctuate, maintaining equilibrium with a reservoir. In GCMC, particle insertions, deletions, and displacements are performed based on probabilities derived from the chemical potential, temperature, and energy states. The acceptance of these moves is determined by the Metropolis criterion, which involves the Boltzmann factor. The thermal de Broglie wavelength, λ, is a quantum mechanical property that relates to the thermal wavelength of particles and is used in calculating the acceptance probability for particle insertions and deletions. The simulation tracks the energy and number of particles over time, providing insights into the system's equilibrium properties.

def GCMC(initial_positions, L, T, mu, sigma, epsilon, mass, num_steps, prob_insertion, prob_deletion, disp_size):



    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K

    # Calculate the thermal de Broglie wavelength
    Lambda = np.sqrt((2 * np.pi * mass * k_B * T) / (h**2))

    # Initialize tracking arrays and counters
    Energy_Trace = []
    Num_particle_Trace = []
    Trail_move_counts_tracker = {'Insertion': [0, 0], 'Deletion': [0, 0], 'Move': [0, 0]}

    # Initial energy of the system
    current_energy = E_system(initial_positions, L, sigma, epsilon)
    current_positions = initial_positions.copy()

    for step in range(num_steps):
        # Decide which move to attempt
        move_type = np.random.choice(['Insertion', 'Deletion', 'Move'], p=[prob_insertion, prob_deletion, 1 - prob_insertion - prob_deletion])

        if move_type == 'Insertion':
            Trail_move_counts_tracker['Insertion'][0] += 1
            # Attempt to insert a new particle
            new_position = np.random.uniform(0, L, 3)
            new_positions = np.append(current_positions, [new_position], axis=0)
            new_energy = E_system(new_positions, L, sigma, epsilon)
            delta_energy = new_energy - current_energy

            # Metropolis criterion for insertion
            acceptance_prob = np.exp(-(delta_energy - mu) / (k_B * T)) * (L**3 / (len(current_positions) + 1) / Lambda**3)
            if np.random.rand() < acceptance_prob:
                current_positions = new_positions
                current_energy = new_energy
                Trail_move_counts_tracker['Insertion'][1] += 1

        elif move_type == 'Deletion' and len(current_positions) > 0:
            Trail_move_counts_tracker['Deletion'][0] += 1
            # Attempt to delete a random particle
            idx_to_remove = np.random.randint(len(current_positions))
            new_positions = np.delete(current_positions, idx_to_remove, axis=0)
            new_energy = E_system(new_positions, L, sigma, epsilon)
            delta_energy = new_energy - current_energy

            # Metropolis criterion for deletion
            acceptance_prob = np.exp(-(delta_energy + mu) / (k_B * T)) * (len(current_positions) / L**3 * Lambda**3)
            if np.random.rand() < acceptance_prob:
                current_positions = new_positions
                current_energy = new_energy
                Trail_move_counts_tracker['Deletion'][1] += 1

        elif move_type == 'Move' and len(current_positions) > 0:
            Trail_move_counts_tracker['Move'][0] += 1
            # Attempt to displace a random particle
            idx_to_move = np.random.randint(len(current_positions))
            displacement = np.random.uniform(-disp_size, disp_size, 3)
            new_position = wrap(current_positions[idx_to_move] + displacement, L)
            new_positions = current_positions.copy()
            new_positions[idx_to_move] = new_position
            new_energy = E_system(new_positions, L, sigma, epsilon)
            delta_energy = new_energy - current_energy

            # Metropolis criterion for move
            acceptance_prob = np.exp(-delta_energy / (k_B * T))
            if np.random.rand() < acceptance_prob:
                current_positions = new_positions
                current_energy = new_energy
                Trail_move_counts_tracker['Move'][1] += 1

        # Record the current state
        Energy_Trace.append(current_energy)
        Num_particle_Trace.append(len(current_positions))

    return Energy_Trace, Num_particle_Trace, Trail_move_counts_tracker, Lambda

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('64.6', 1)
target = targets[0]

def initialize_fcc(N,spacing = 1.3):
    ## this follows HOOMD tutorial ##
    K = int(np.ceil(N ** (1 / 3)))
    L = K * spacing
    x = np.linspace(-L/2, L/2, K, endpoint=False)
    position = list(itertools.product(x, repeat=3))
    return [np.array(position),L]
mass = 1
sigma = 1.0
epsilon = 0
mu = 1.0
T = 1.0
N = 64
num_steps = int(1e5)
init_positions, L = initialize_fcc(N, spacing = 0.2)
np.random.seed(0)
Energy_Trace, Num_particle_Trace, Tracker, Lambda = GCMC(init_positions, L, T, mu, sigma, epsilon, mass, num_steps,
                                        prob_insertion = 0.3, prob_deletion = 0.3, disp_size = 0.5 )
assert (abs(np.average(Num_particle_Trace[40000:])-np.exp(mu/T)*(L/Lambda)**3)/(np.exp(mu/T)*(L/Lambda)**3)< 0.01) == target
