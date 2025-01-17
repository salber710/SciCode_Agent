import numpy as np
import itertools

# Background: 
# In quantum mechanics, the ground state energy of a particle in a 1D infinite square well is given by the formula:
# E_n = (n^2 * h^2) / (8 * m * L^2), where n is the quantum number (n=1 for ground state), h is the Planck constant,
# m is the mass of the particle, and L is the width of the well. For the ground state, n=1.
# The effective mass m_r is used to account for the mass of the electron in a material, and it is given as a 
# multiple of the free electron mass m_0. Therefore, the effective mass m = m_r * m_0.
# The energy of a photon is related to its wavelength by the equation E = h * c / λ, where c is the speed of light.
# By equating the ground state energy to the photon energy, we can solve for the wavelength λ.

def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    # Constants
    h = 6.626e-34  # Planck constant in J*s
    c = 3e8        # Speed of light in m/s
    m0 = 9.109e-31 # Free electron mass in kg

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Calculate the effective mass
    m = mr * m0

    # Calculate the ground state energy E1
    E1 = (h**2) / (8 * m * L_m**2)

    # Calculate the wavelength λ corresponding to the ground state energy
    lmbd = (h * c) / E1

    # Convert the wavelength from meters to nanometers
    lmbd_nm = lmbd * 1e9

    return lmbd_nm


# Background: In mathematics, a quadratic combination of numbers involves expressions where each number is multiplied by the square of an integer coefficient. 
# For three given numbers x, y, and z, a quadratic combination can be expressed as i^2 * x + j^2 * y + k^2 * z, where i, j, and k are non-negative integers. 
# The task is to generate the smallest N such combinations in ascending order. This involves iterating over possible values of i, j, and k, calculating the 
# quadratic combination, and collecting the smallest unique results. The itertools library can be used to generate combinations of indices efficiently.



def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    # Set to store unique quadratic combinations
    combinations = set()
    
    # Iterate over possible values of i, j, k
    # We use a reasonable range for i, j, k to ensure we get at least N combinations
    # The range is chosen based on the assumption that the smallest combinations will be formed by small i, j, k
    max_range = int(np.ceil(np.sqrt(N))) + 1  # A heuristic to limit the range of i, j, k
    
    for i in range(max_range):
        for j in range(max_range):
            for k in range(max_range):
                # Calculate the quadratic combination
                combination = i**2 * x + j**2 * y + k**2 * z
                # Add to the set to ensure uniqueness
                combinations.add(combination)
    
    # Convert the set to a sorted list
    sorted_combinations = sorted(combinations)
    
    # Return the first N elements as a numpy array
    return np.array(sorted_combinations[:N])



# Background: In quantum mechanics, the energy levels of a particle in a 3D infinite potential well (cuboid quantum dot) are quantized.
# The energy levels are given by the formula: E_nx,ny,nz = (h^2 / 8 * m) * (nx^2 / a^2 + ny^2 / b^2 + nz^2 / c^2),
# where nx, ny, nz are quantum numbers (positive integers), h is the Planck constant, m is the effective mass of the particle,
# and a, b, c are the dimensions of the cuboid. The task is to calculate the smallest N non-zero energy levels by considering
# all possible combinations of nx, ny, nz, and then return these energy levels in descending order.



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
    h = 6.626e-34  # Planck constant in J*s
    m0 = 9.109e-31 # Free electron mass in kg

    # Convert dimensions from nanometers to meters
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9

    # Calculate the effective mass
    m = mr * m0

    # Set to store unique energy levels
    energy_levels = set()

    # Iterate over possible values of nx, ny, nz
    # We use a reasonable range for nx, ny, nz to ensure we get at least N energy levels
    max_range = int(np.ceil(np.sqrt(N))) + 1  # A heuristic to limit the range of nx, ny, nz

    for nx in range(1, max_range):
        for ny in range(1, max_range):
            for nz in range(1, max_range):
                # Calculate the energy level
                energy = (h**2 / (8 * m)) * ((nx**2 / a_m**2) + (ny**2 / b_m**2) + (nz**2 / c_m**2))
                # Add to the set to ensure uniqueness
                energy_levels.add(energy)

    # Convert the set to a sorted list in descending order
    sorted_energy_levels = sorted(energy_levels, reverse=True)

    # Return the first N elements as a numpy array
    return np.array(sorted_energy_levels[:N])


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
