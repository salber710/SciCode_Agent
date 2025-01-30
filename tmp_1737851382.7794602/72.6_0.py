import numpy as np

# Background: In a lattice model, each site has a set of nearest neighbors. For a 2D lattice with periodic boundary conditions, 
# the neighbors of a site (i, j) are determined by considering the lattice as a torus. This means that if a site is at the edge 
# of the lattice, its neighbors wrap around to the opposite edge. For a site (i, j) in an N x N lattice, the neighbors are:
# (i-1, j) - the site above, (i, j+1) - the site to the right, (i+1, j) - the site below, and (i, j-1) - the site to the left.
# The periodic boundary conditions ensure that these indices wrap around using modulo N arithmetic.

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    if not isinstance(N, int):
        raise TypeError("N must be an integer")
    if not isinstance(site, tuple) or not len(site) == 2 or not all(isinstance(x, int) for x in site):
        raise TypeError("site must be a tuple of two integers")
    if any(x >= N or x < 0 for x in site):
        raise IndexError("site indices must be within the range of the lattice size")
    if N <= 0:
        raise IndexError("N must be a positive integer")

    i, j = site
    # Calculate neighbors with periodic boundary conditions
    i_above = (i - 1) % N
    j_right = (j + 1) % N
    i_below = (i + 1) % N
    j_left = (j - 1) % N
    
    # Return the list of neighbors
    return [(i, j_left), (i_above, j), (i, j_right), (i_below, j)]


# Background: In the Ising model, the energy of a particular site in a lattice is determined by its interaction with its nearest neighbors.
# The energy contribution of a site (i, j) is calculated using the formula: E_ij = -S_ij * (S_neighbors_sum), where S_ij is the spin at site (i, j),
# and S_neighbors_sum is the sum of the spins of its nearest neighbors. The negative sign indicates that aligned spins (same sign) lower the energy,
# which is a characteristic of ferromagnetic interactions. The periodic boundary conditions ensure that the lattice behaves like a torus, 
# meaning that spins on the edges interact with those on the opposite edges.


def energy_site(i, j, lattice):
    '''Calculate the energy of site (i, j)
    Args:
        i (int): site index along x
        j (int): site index along y
        lattice (np.array): shape (N, N), a 2D array of +1 and -1
    Return:
        float: energy of site (i, j)
    '''
    N = lattice.shape[0]
    
    # Validate the lattice values
    if not np.all(np.isin(lattice, [1, -1])):
        raise ValueError("Lattice should only contain values 1 or -1.")
    
    # Validate indices
    if not (0 <= i < N and 0 <= j < N):
        raise IndexError("Index (i, j) out of bounds.")
    
    # Get the spin at the current site
    S_ij = lattice[i, j]
    
    # Calculate the indices of the neighbors using periodic boundary conditions
    i_above = (i - 1) % N
    j_right = (j + 1) % N
    i_below = (i + 1) % N
    j_left = (j - 1) % N
    
    # Sum the spins of the nearest neighbors
    S_neighbors_sum = (lattice[i_above, j] + lattice[i, j_right] +
                       lattice[i_below, j] + lattice[i, j_left])
    
    # Calculate the energy of the site
    energy = -S_ij * S_neighbors_sum
    
    return energy


# Background: In the Ising model, the total energy of the system is the sum of the energies of all individual sites.
# Each site's energy is determined by its interaction with its nearest neighbors, calculated as E_ij = -S_ij * (S_neighbors_sum).
# Due to periodic boundary conditions, the lattice behaves like a torus, meaning that spins on the edges interact with those on the opposite edges.
# To calculate the total energy of the lattice, we sum the energy contributions of all sites. However, since each pair of neighboring spins
# is counted twice (once for each site in the pair), the total energy should be divided by 2 to avoid double-counting.


def energy(lattice):
    '''Calculate the total energy for the site (i, j) of the periodic Ising model with dimension (N, N)
    Args:
        lattice (np.array): shape (N, N), a 2D array of +1 and -1
    Return:
        float: total energy
    '''
    N = lattice.shape[0]
    
    # Validate the lattice values
    if not np.all(np.isin(lattice, [1, -1])):
        raise ValueError("Lattice should only contain values 1 or -1.")
    
    # Validate if lattice is empty or not properly shaped
    if N == 0 or lattice.shape[1] != N:
        raise ValueError("Lattice must be a non-empty square matrix.")
    
    total_energy = 0.0
    
    # Iterate over each site in the lattice
    for i in range(N):
        for j in range(N):
            # Get the spin at the current site
            S_ij = lattice[i, j]
            
            # Calculate the indices of the neighbors using periodic boundary conditions
            i_above = (i - 1) % N
            j_right = (j + 1) % N
            i_below = (i + 1) % N
            j_left = (j - 1) % N
            
            # Sum the spins of the nearest neighbors
            S_neighbors_sum = lattice[i_above, j] + lattice[i, j_right] + lattice[i_below, j] + lattice[i, j_left]
            
            # Calculate the energy contribution of the site and add to total energy
            total_energy += -S_ij * S_neighbors_sum
    
    # Since each pair of neighboring spins is counted twice, divide the total energy by 2
    total_energy /= 2
    
    # Special case for single element lattice
    if N == 1:
        total_energy = 0.0
    
    return total_energy


# Background: In the Ising model, the magnetization of a lattice is a measure of the net magnetic moment of the system.
# It is calculated as the sum of all the spins in the lattice. Each spin can be either +1 or -1, representing the two possible
# states of a magnetic dipole. The total magnetization is an important quantity as it indicates the degree of alignment of the spins.
# A positive magnetization indicates a net alignment in the positive direction, while a negative magnetization indicates a net alignment
# in the negative direction. In a perfectly balanced system with equal numbers of +1 and -1 spins, the magnetization would be zero.


def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: total magnetization
    '''
    # Validate the spins array
    if not spins.ndim == 2:
        raise ValueError("Spins should be a 2D array.")
    if not np.all(np.isin(spins, [1, -1])):
        raise ValueError("Spins should only contain values 1 or -1.")
    if spins.size == 0:
        raise ValueError("Spins array should not be empty.")
    
    # Calculate the total magnetization by summing all the spins
    mag = np.sum(spins)
    
    return mag


# Background: In the Ising model, the acceptance probability for a spin flip is determined by the change in energy
# that the flip would cause. The Metropolis criterion is often used, where the acceptance probability A is given by
# A = min(1, exp(-beta * delta_E)), where delta_E is the change in energy due to the spin flip, and beta is the inverse
# temperature. The change in magnetization, dM, is simply twice the value of the spin at the site (i, j) because flipping
# the spin changes its contribution to the total magnetization by this amount.


def get_flip_probability_magnetization(lattice, i, j, beta):
    '''Calculate spin flip probability and change in total magnetization.
    Args:
        lattice (np.array): shape (N, N), 2D lattice of 1 and -1
        i (int): site index along x
        j (int): site index along y
        beta (float): inverse temperature
    Return:
        A (float): acceptance ratio
        dM (int): change in magnetization after the spin flip
    '''
    N = lattice.shape[0]
    
    # Validate the lattice values
    if not np.all(np.isin(lattice, [1, -1])):
        raise ValueError("Lattice should only contain values 1 or -1.")
    
    # Validate indices
    if not (0 <= i < N and 0 <= j < N):
        raise IndexError("Index (i, j) out of bounds.")
    
    # Validate lattice shape
    if lattice.shape[0] != lattice.shape[1]:
        raise ValueError("Lattice must be square (N x N).")
    
    # Validate beta
    if beta < 0:
        raise ValueError("Beta must be non-negative.")
    
    # Get the spin at the current site
    S_ij = lattice[i, j]
    
    # Calculate the indices of the neighbors using periodic boundary conditions
    i_above = (i - 1) % N
    j_right = (j + 1) % N
    i_below = (i + 1) % N
    j_left = (j - 1) % N
    
    # Sum the spins of the nearest neighbors
    S_neighbors_sum = (lattice[i_above, j] + lattice[i, j_right] +
                       lattice[i_below, j] + lattice[i, j_left])
    
    # Calculate the change in energy due to the spin flip
    delta_E = 2 * S_ij * S_neighbors_sum
    
    # Calculate the acceptance probability using the Metropolis criterion
    A = min(1, np.exp(-beta * delta_E))
    
    # Calculate the change in magnetization
    dM = -2 * S_ij
    
    return A, dM



# Background: In the Ising model, the Metropolis algorithm is used to simulate the system's evolution towards equilibrium.
# The algorithm involves iterating over each spin in the lattice and deciding whether to flip it based on the Metropolis criterion.
# For each spin, the acceptance probability for a flip is calculated using the function `get_flip_probability_magnetization()`.
# A random number is generated, and if this number is less than the acceptance probability, the spin is flipped.
# This process is repeated for all spins in the lattice, allowing the system to explore different configurations and
# eventually reach a state that reflects the thermal equilibrium at the given temperature.


def flip(spins, beta):
    '''Goes through each spin in the 2D lattice and flip it.
    Args:
        spins (np.array): shape (N, N), 2D lattice of 1 and -1        
        beta (float): inverse temperature
    Return:
        lattice (np.array): final spin configurations
    '''
    N = spins.shape[0]
    
    # Iterate over each spin in the lattice
    for i in range(N):
        for j in range(N):
            # Calculate the acceptance probability and magnetization change
            A, dM = get_flip_probability_magnetization(spins, i, j, beta)
            
            # Generate a random number between 0 and 1
            random_number = np.random.rand()
            
            # Flip the spin if the random number is less than the acceptance probability
            if random_number < A:
                spins[i, j] *= -1  # Flip the spin
    
    return spins

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('72.6', 3)
target = targets[0]

np.random.seed(0)
spins = np.array([[ 1, -1,  1,  1],[-1, -1,  1,  1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
assert np.allclose(flip(spins, 1), target)
target = targets[1]

np.random.seed(1)
spins = np.array([[ 1, -1,  1,  1],[-1, -1,  -1,  1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
assert np.allclose(flip(spins, 1), target)
target = targets[2]

np.random.seed(2)
spins = np.array([[ 1, -1,  1,  1],[-1, -1,  1,  -1],[-1, -1,  1,  1],[ 1, -1, -1, -1]])
assert np.allclose(flip(spins, 1), target)
