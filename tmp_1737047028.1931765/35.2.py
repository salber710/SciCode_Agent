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
