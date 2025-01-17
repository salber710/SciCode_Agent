import numpy as np

# Background: In a lattice model, each site can interact with its nearest neighbors. For a 2D lattice with periodic boundary conditions, 
# the lattice "wraps around" such that the edges are connected. This means that for a site at the edge of the lattice, 
# its neighbors include sites on the opposite edge. For a site (i, j) in an N x N lattice, the neighbors are:
# - Left neighbor: (i, j-1) which wraps to (i, N-1) if j is 0
# - Right neighbor: (i, j+1) which wraps to (i, 0) if j is N-1
# - Above neighbor: (i-1, j) which wraps to (N-1, j) if i is 0
# - Below neighbor: (i+1, j) which wraps to (0, j) if i is N-1

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    # Calculate neighbors with periodic boundary conditions
    left = (i, (j - 1) % N)
    right = (i, (j + 1) % N)
    above = ((i - 1) % N, j)
    below = ((i + 1) % N, j)
    
    return [left, above, right, below]


# Background: In the Ising model, the energy of a particular site in a lattice is determined by its interaction with its nearest neighbors.
# The energy of a site (i, j) is given by the formula: E_ij = -S_ij * (S_left + S_right + S_above + S_below), where S_ij is the spin at site (i, j),
# and S_left, S_right, S_above, S_below are the spins of its nearest neighbors. The negative sign indicates that aligned spins (same sign) lower the energy,
# which is a characteristic of ferromagnetic interactions. The total energy of the site is the sum of these interactions.


def energy_site(i, j, lattice):
    '''Calculate the energy of site (i, j)
    Args:
        i (int): site index along x
        j (int): site index along y
        lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy of site (i, j)
    '''
    N = lattice.shape[0]
    # Get the spin of the current site
    S_ij = lattice[i, j]
    
    # Calculate the indices of the neighbors using periodic boundary conditions
    left = lattice[i, (j - 1) % N]
    right = lattice[i, (j + 1) % N]
    above = lattice[(i - 1) % N, j]
    below = lattice[(i + 1) % N, j]
    
    # Calculate the energy contribution from the site (i, j)
    energy = -S_ij * (left + right + above + below)
    
    return energy


# Background: In the Ising model, the total energy of the system is the sum of the energies of all individual sites.
# Each site contributes to the energy based on its interaction with its nearest neighbors, as described by the formula:
# E_ij = -S_ij * (S_left + S_right + S_above + S_below). To calculate the total energy of the lattice, we sum the energy
# contributions from all sites. However, since each pair of neighboring sites is counted twice (once for each site),
# we divide the total sum by 2 to avoid double-counting.


def energy(lattice):
    '''Calculate the total energy for the site (i, j) of the periodic Ising model with dimension (N, N)
    Args:
        lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy 
    '''
    N = lattice.shape[0]
    total_energy = 0.0
    
    for i in range(N):
        for j in range(N):
            S_ij = lattice[i, j]
            # Calculate the indices of the neighbors using periodic boundary conditions
            left = lattice[i, (j - 1) % N]
            right = lattice[i, (j + 1) % N]
            above = lattice[(i - 1) % N, j]
            below = lattice[(i + 1) % N, j]
            
            # Calculate the energy contribution from the site (i, j)
            site_energy = -S_ij * (left + right + above + below)
            total_energy += site_energy
    
    # Each pair of neighbors is counted twice, so divide by 2
    total_energy /= 2.0
    
    return total_energy


# Background: In the Ising model, the magnetization of a lattice is a measure of the net magnetic moment of the system.
# It is calculated as the sum of all the spins in the lattice. Each spin can be either +1 or -1, representing the two possible
# states of a magnetic moment. The total magnetization is simply the sum of these values across the entire lattice.
# A positive magnetization indicates a net alignment of spins in the positive direction, while a negative magnetization
# indicates a net alignment in the negative direction. If the magnetization is zero, it suggests that the spins are
# equally distributed between the two states, resulting in no net magnetic moment.

def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: 
    '''
    # Calculate the total magnetization by summing all the spins in the lattice
    mag = np.sum(spins)
    
    return mag


# Background: In the Ising model, the acceptance probability for a spin flip is determined by the change in energy
# due to the flip and the inverse temperature, beta. The change in energy, ΔE, is calculated as the difference in energy
# before and after the spin flip. The acceptance probability, A, is given by the Metropolis criterion: A = min(1, exp(-beta * ΔE)).
# If ΔE is negative, the flip is energetically favorable and A = 1. If ΔE is positive, the flip is accepted with probability
# exp(-beta * ΔE). The change in magnetization, ΔM, is simply twice the value of the spin at (i, j) because flipping the spin
# changes its contribution to the total magnetization from +1 to -1 or vice versa.


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
    S_ij = lattice[i, j]
    
    # Calculate the indices of the neighbors using periodic boundary conditions
    left = lattice[i, (j - 1) % N]
    right = lattice[i, (j + 1) % N]
    above = lattice[(i - 1) % N, j]
    below = lattice[(i + 1) % N, j]
    
    # Calculate the change in energy ΔE due to the spin flip
    delta_E = 2 * S_ij * (left + right + above + below)
    
    # Calculate the acceptance probability using the Metropolis criterion
    A = min(1, np.exp(-beta * delta_E))
    
    # Calculate the change in magnetization ΔM
    dM = -2 * S_ij
    
    return A, dM



# Background: In the Ising model, the Metropolis algorithm is used to simulate the evolution of the system towards equilibrium.
# The algorithm involves iterating over each spin in the lattice, calculating the acceptance probability for flipping the spin
# using the Metropolis criterion, and then deciding whether to flip the spin based on a random number. If the random number
# is less than the acceptance probability, the spin is flipped, indicating that the new configuration is accepted. This process
# is repeated for all spins in the lattice, allowing the system to explore different configurations and eventually reach a state
# of equilibrium. The acceptance probability is determined by the change in energy due to the spin flip and the inverse temperature.


def flip(spins, beta):
    '''Goes through each spin in the 2D lattice and flip it.
    Args:
        spins (np.array): shape (N, N), 2D lattice of 1 and -1        
        beta (float): inverse temperature
    Return:
        lattice (np.array): final spin configurations
    '''
    N = spins.shape[0]
    for i in range(N):
        for j in range(N):
            # Calculate the acceptance probability and magnetization change
            A, _ = get_flip_probability_magnetization(spins, i, j, beta)
            
            # Generate a random number and decide whether to flip the spin
            if np.random.rand() < A:
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
