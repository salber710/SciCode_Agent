from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j).
    Args:
        site (Tuple[int, int]): site indices
        N (int): number of sites along each dimension
    Return:
        list: a list of 2-tuples, [(i_left, j_left), (i_above, j_above), (i_right, j_right), (i_below, j_below)]
    '''
    i, j = site
    
    # Define a custom function to handle periodic boundary conditions using bitwise operations
    def wrap_with_bitwise(x, max_val):
        return x & (max_val - 1)
    
    # Compute neighbors using the custom wrapping function
    neighbors = [
        (wrap_with_bitwise(i - 1 + N, N), j),  # left
        (i, wrap_with_bitwise(j + 1 + N, N)),  # above
        (wrap_with_bitwise(i + 1 + N, N), j),  # right
        (i, wrap_with_bitwise(j - 1 + N, N))   # below
    ]
    
    return neighbors



def neighbor_list(site, N):
    '''Return all nearest neighbors of site (i, j) using a different approach to handle periodic conditions.'''
    i, j = site

    # Using a dictionary to calculate neighbors with periodic boundary condition
    neighbors = {
        'left': ((i - 1 + N) % N, j),
        'above': (i, (j + 1 + N) % N),
        'right': ((i + 1) % N, j),
        'below': (i, (j - 1 + N) % N)
    }

    return neighbors.values()

def energy_site(i, j, lattice):
    '''Calculate the energy of site (i, j) using numpy broadcasting for neighbor interaction.
    Args:
        i (int): site index along x
        j (int): site index along y
        lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy of site (i, j)
    '''
    N = lattice.shape[0]
    spin = lattice[i, j]

    # Get neighbors using the dictionary values
    neighbors = neighbor_list((i, j), N)

    # Calculate energy using numpy broadcasting and array operations
    neighbor_spins = np.array([lattice[ni, nj] for ni, nj in neighbors])
    energy = -spin * np.dot(neighbor_spins, [1, 1, 1, 1])

    return energy



def energy(lattice):
    '''calculate the total energy for the site (i, j) of the periodic Ising model with dimension (N, N)
    Args: lattice (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: energy 
    '''
    N = lattice.shape[0]
    total_energy = 0
    
    # Consider diagonal neighbors for variety, even though they don't contribute in the standard model
    for i in range(N):
        for j in range(N):
            spin = lattice[i, j]
            # Calculate energy by considering right and down neighbors
            right_neighbor_energy = spin * lattice[i, (j + 1) % N]
            down_neighbor_energy = spin * lattice[(i + 1) % N, j]
            
            # Add the contributions to the total energy
            total_energy += right_neighbor_energy + down_neighbor_energy
    
    # Return the negative of the accumulated energy to match the typical Ising model convention
    return -total_energy


def magnetization(spins):
    '''total magnetization of the periodic Ising model with dimension (N, N)
    Args: spins (np.array): shape (N, N), a 2D array +1 and -1
    Return:
        float: 
    '''
    mag = 0
    for index, value in np.ndenumerate(spins):
        mag += value
    return float(mag)



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
    spin = lattice[i, j]

    # Calculate neighbor offsets using a different ordering
    neighbor_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # Calculate ΔE using a loop and explicit coordinate wrapping
    delta_E = 0
    for offset in neighbor_offsets:
        ni, nj = (i + offset[0]) % N, (j + offset[1]) % N
        delta_E += lattice[ni, nj]
    delta_E *= 2 * spin

    # Use a cosine function to calculate acceptance probability
    A = 1 / (1 + np.cos(beta * delta_E))

    # Magnetization change
    dM = -2 * spin

    return A, dM



def flip(spins, beta):
    '''Goes through each spin in the 2D lattice using a concentric square pattern and flips it based on acceptance probability.
    Args:
        spins (np.array): shape (N, N), 2D lattice of 1 and -1
        beta (float): inverse temperature
    Return:
        lattice (np.array): final spin configurations
    '''
    N = spins.shape[0]
    
    # Process spins in concentric squares starting from the outermost
    layers = (N + 1) // 2
    for layer in range(layers):
        # Top row, left to right
        for j in range(layer, N - layer):
            i = layer
            A, dM = get_flip_probability_magnetization(spins, i, j, beta)
            if np.random.uniform() < A:
                spins[i, j] *= -1
        
        # Right column, top to bottom
        for i in range(layer + 1, N - layer):
            j = N - layer - 1
            A, dM = get_flip_probability_magnetization(spins, i, j, beta)
            if np.random.uniform() < A:
                spins[i, j] *= -1
        
        # Bottom row, right to left
        if N - layer - 1 > layer:  # Avoid double-processing the same row
            for j in range(N - layer - 2, layer - 1, -1):
                i = N - layer - 1
                A, dM = get_flip_probability_magnetization(spins, i, j, beta)
                if np.random.uniform() < A:
                    spins[i, j] *= -1
        
        # Left column, bottom to top
        if layer < N - layer - 1:  # Avoid double-processing the same column
            for i in range(N - layer - 2, layer, -1):
                j = layer
                A, dM = get_flip_probability_magnetization(spins, i, j, beta)
                if np.random.uniform() < A:
                    spins[i, j] *= -1

    return spins



def run(T, N, nsweeps):
    '''Performs Metropolis to flip spins for nsweeps times and collect magnetization^2 / N^4.
    Args:
        T (float): temperature
        N (int): system size along an axis
        nsweeps: number of iterations to go over all spins
    Return:
        mag2: (numpy array) magnetization^2 / N^4
    '''
    
    # Initialize the lattice with spins in a random configuration
    lattice = np.where(np.random.rand(N, N) < 0.5, -1, 1)
    beta = 1.0 / T

    # Helper function to compute the energy change for a spin flip
    def compute_energy_change(x, y):
        return 2 * lattice[x, y] * (
            lattice[(x + 1) % N, y] + lattice[(x - 1) % N, y] +
            lattice[x, (y + 1) % N] + lattice[x, (y - 1) % N]
        )

    mag2_list = []

    for sweep in range(nsweeps):
        # Perform a random walk visiting each spin once
        x, y = np.random.randint(0, N), np.random.randint(0, N)
        for _ in range(N * N):
            dE = compute_energy_change(x, y)

            # Metropolis acceptance criterion
            if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                lattice[x, y] *= -1

            # Move to a random neighboring spin
            x, y = (x + np.random.choice([-1, 1])) % N, (y + np.random.choice([-1, 1])) % N

        # Calculate the magnetization and its square
        total_magnetization = np.sum(lattice)
        mag2_list.append((total_magnetization ** 2) / (N ** 4))

    return np.array(mag2_list)


def scan_T(Ts, N, nsweeps):
    '''Run over several given temperatures.
    Args:
        Ts: list of temperatures
        N: system size in one axis
        nsweeps: number of iterations to go over all spins
    Return:
        mag2_avg: list of magnetization^2 / N^4, each element is the value for each temperature
    '''


    def checkerboard_step(lattice, beta):
        '''Perform updates on a checkerboard pattern.'''
        # Update black sites
        for i in range(N):
            for j in range(N):
                if (i + j) % 2 == 0:  # Black site
                    sum_neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                                     lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
                    delta_E = 2 * lattice[i, j] * sum_neighbors
                    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                        lattice[i, j] *= -1

        # Update white sites
        for i in range(N):
            for j in range(N):
                if (i + j) % 2 == 1:  # White site
                    sum_neighbors = (lattice[(i+1)%N, j] + lattice[(i-1)%N, j] +
                                     lattice[i, (j+1)%N] + lattice[i, (j-1)%N])
                    delta_E = 2 * lattice[i, j] * sum_neighbors
                    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                        lattice[i, j] *= -1

    def calculate_magnetization_squared(lattice):
        '''Calculate the magnetization squared of the lattice.'''
        total_magnetization = np.sum(lattice)
        return (total_magnetization ** 2)

    mag2_avg = []

    for T in Ts:
        beta = 1.0 / T
        lattice = np.random.choice([-1, 1], size=(N, N))
        
        magnetization_squared_sum = 0.0
        
        for _ in range(nsweeps):
            checkerboard_step(lattice, beta)
            magnetization_squared_sum += calculate_magnetization_squared(lattice)
        
        mag2_avg.append(magnetization_squared_sum / (nsweeps * N ** 4))
    
    return mag2_avg



def calc_transition(T_list, mag2_list):
    '''Calculates the transition temperature by taking derivative
    Args:
        T_list: list of temperatures
        mag2_list: list of magnetization^2/N^4 at each temperature
    Return:
        float: Transition temperature
    '''
    # Calculate the central differences for the derivative
    derivatives = [(mag2_list[i + 1] - mag2_list[i]) / (T_list[i + 1] - T_list[i]) for i in range(len(T_list) - 1)]

    # Handle case for the last temperature by repeating the previous derivative
    derivatives.append(derivatives[-1])

    # Find the index where the derivative value is minimized
    min_derivative_index = derivatives.index(min(derivatives))

    # Return the temperature corresponding to this index as the transition temperature
    T_transition = (T_list[min_derivative_index] + T_list[min_derivative_index + 1]) / 2

    return T_transition


try:
    targets = process_hdf5_to_tuple('72.9', 4)
    target = targets[0]
    np.random.seed(0)
    Ts = [1.6, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.8]
    mag2 = scan_T(Ts=Ts, N=5, nsweeps=100)
    assert np.allclose(calc_transition(Ts, mag2), target)

    target = targets[1]
    np.random.seed(0)
    Ts = [1.6, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.8]
    mag2 = scan_T(Ts=Ts, N=10, nsweeps=100)
    assert np.allclose(calc_transition(Ts, mag2), target)

    target = targets[2]
    np.random.seed(0)
    Ts = [1.6, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.8]
    mag2 = scan_T(Ts=Ts, N=20, nsweeps=100)
    assert np.allclose(calc_transition(Ts, mag2), target)

    target = targets[3]
    np.random.seed(0)
    Ts = [1.6, 2.10, 2.15, 2.20, 2.25, 2.30, 2.35, 2.40, 2.8]
    mag2 = scan_T(Ts=Ts, N=30, nsweeps=2000)
    T_transition = calc_transition(Ts, mag2)
    assert (np.abs(T_transition - 2.269) < 0.2) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e