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
    # Initialize a set to store unique quadratic combinations
    combinations = set()
    
    # We will iterate over possible values of i, j, k
    # The range is chosen arbitrarily large to ensure we find at least N combinations
    # This can be optimized further based on specific constraints or requirements
    max_range = 100  # This is a heuristic choice; adjust as needed for larger N
    
    for i in range(max_range):
        for j in range(max_range):
            for k in range(max_range):
                # Calculate the quadratic combination
                combination = i**2 * x + j**2 * y + k**2 * z
                # Add the combination to the set
                combinations.add(combination)
                # If we have enough combinations, break early
                if len(combinations) >= N:
                    break
            if len(combinations) >= N:
                break
        if len(combinations) >= N:
            break
    
    # Convert the set to a sorted list
    sorted_combinations = sorted(combinations)
    
    # Return the first N elements as a numpy array
    return np.array(sorted_combinations[:N])

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.2', 3)
target = targets[0]

C = generate_quadratic_combinations(7, 11, 13, 5)
assert np.allclose(sorted(C), target)
target = targets[1]

C = generate_quadratic_combinations(7, 11, 13, 10)
assert np.allclose(sorted(C), target)
target = targets[2]

C = generate_quadratic_combinations(71, 19, 17, 5)
assert np.allclose(sorted(C), target)
