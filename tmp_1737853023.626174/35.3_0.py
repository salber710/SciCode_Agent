import numpy as np
import itertools

# Background: In quantum mechanics, the ground state energy of a particle in a 1D infinite square well is given by the formula:
# E = (h^2 * n^2) / (8 * m * L^2), where h is the Planck constant, n is the principal quantum number (n=1 for ground state),
# m is the effective mass of the particle, and L is the width of the well. The effective mass m is given by m_r * m_e,
# where m_r is the relative effective mass and m_e is the free electron mass. The energy E can be related to the wavelength
# of a photon by the equation E = h * c / λ, where c is the speed of light and λ is the wavelength. By equating these two
# expressions for energy, we can solve for the wavelength λ.

def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    if L <= 0:
        raise ValueError("Width L must be positive and non-zero.")
    if mr <= 0:
        raise ValueError("Relative effective mass mr must be positive and non-zero.")

    # Constants
    h = 6.62607015e-34  # Planck constant in J*s
    c = 299792458       # Speed of light in m/s (exact value)
    m_e = 9.10938356e-31  # Free electron mass in kg

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Calculate the effective mass
    m = mr * m_e

    # Calculate the ground state energy E
    E = (h**2) / (8 * m * L_m**2)

    # Calculate the wavelength λ using E = h * c / λ
    lmbd = h * c / E

    # Convert the wavelength from meters to nanometers
    lmbd_nm = lmbd * 1e9

    return lmbd_nm


# Background: In mathematics, a quadratic combination of numbers involves expressions where each number is multiplied by the square of an integer coefficient. 
# For three given numbers x, y, and z, a quadratic combination can be expressed as i^2 * x + j^2 * y + k^2 * z, where i, j, and k are non-negative integers. 
# The task is to generate the smallest N such combinations in ascending order. This involves iterating over possible values of i, j, and k, computing the 
# quadratic combination, and collecting the smallest N unique results. The itertools library can be used to generate combinations of indices efficiently.




def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    # Initialize a min-heap to store the quadratic combinations
    min_heap = []
    
    # Use a set to avoid duplicate combinations
    seen = set()
    
    # Start with the smallest possible combination (0,0,0)
    heapq.heappush(min_heap, (0, 0, 0, 0))  # (value, i, j, k)
    seen.add((0, 0, 0))
    
    # Result list to collect the smallest N combinations
    result = []
    
    # Extract the smallest elements from the heap until we have N elements
    while len(result) < N:
        value, i, j, k = heapq.heappop(min_heap)
        result.append(value)
        
        # Generate new combinations by incrementing i, j, k
        if (i+1, j, k) not in seen:
            seen.add((i+1, j, k))
            heapq.heappush(min_heap, ((i+1)**2 * x + j**2 * y + k**2 * z, i+1, j, k))
        
        if (i, j+1, k) not in seen:
            seen.add((i, j+1, k))
            heapq.heappush(min_heap, (i**2 * x + (j+1)**2 * y + k**2 * z, i, j+1, k))
        
        if (i, j, k+1) not in seen:
            seen.add((i, j, k+1))
            heapq.heappush(min_heap, (i**2 * x + j**2 * y + (k+1)**2 * z, i, j, k+1))
    
    # Return the first N elements as a numpy array
    return np.array(result)



# Background: In quantum mechanics, the energy levels of a particle confined in a 3D cuboid quantum dot can be calculated using the formula:
# E_nx_ny_nz = (h^2 / (8 * m * L^2)) * (nx^2/a^2 + ny^2/b^2 + nz^2/c^2), where h is the Planck constant, m is the effective mass of the particle,
# L is the characteristic length (in this case, the dimensions a, b, and c of the cuboid), and nx, ny, nz are the quantum numbers for each dimension.
# The effective mass m is given by m_r * m_e, where m_r is the relative effective mass and m_e is the free electron mass.
# The task is to calculate the smallest N non-zero energy levels by considering all possible combinations of nx, ny, and nz, and return them in descending order.



def absorption(mr, a, b, c, N):
    '''With the feature sizes in three dimensions a, b, and c, the relative mass mr and the array length N, return a numpy array of the size N that contains the corresponding photon wavelength of the excited states' energy.
    Input:
    mr (float): relative effective electron mass.
    a (float): Feature size in the first dimension (nm).
    b (float): Feature size in the second dimension (nm).
    c (float): Feature size in the Third dimension (nm).
    N (int): The length of returned array.
    Output:
    A (size N numpy array): The collection of the energy level wavelength.
    '''
    # Constants
    h = 6.62607015e-34  # Planck constant in J*s
    m_e = 9.10938356e-31  # Free electron mass in kg

    # Convert dimensions from nanometers to meters
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9

    # Calculate the effective mass
    m = mr * m_e

    # Calculate the energy prefactor
    prefactor = h**2 / (8 * m)

    # Generate all possible combinations of nx, ny, nz starting from 1
    energy_levels = []
    for nx, ny, nz in itertools.product(range(1, N+1), repeat=3):
        # Calculate the energy for the given combination of nx, ny, nz
        E = prefactor * ((nx**2 / a_m**2) + (ny**2 / b_m**2) + (nz**2 / c_m**2))
        energy_levels.append(E)

    # Sort the energy levels and take the smallest N non-zero levels
    energy_levels = sorted(set(energy_levels))[:N]

    # Return the energy levels in descending order
    return np.array(sorted(energy_levels, reverse=True))

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.3', 4)
target = targets[0]

A = absorption(0.6,3,4,10**6,5)
assert (all(i>10**10 for i in A)) == target
target = targets[1]

A = absorption(0.3,7,3,5,10)
assert np.allclose(sorted(A)[::-1], target)
target = targets[2]

A = absorption(0.6,3,4,5,5)
assert np.allclose(sorted(A)[::-1], target)
target = targets[3]

A = absorption(0.6,37,23,18,10)
assert np.allclose(sorted(A)[::-1], target)
