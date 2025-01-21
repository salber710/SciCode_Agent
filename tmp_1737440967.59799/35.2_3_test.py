import numpy as np
import itertools

# Background: The ground state energy of a particle in a 1D infinite square well is given by the formula:
# E = (h^2 * n^2) / (8 * m * L^2), where h is the Planck constant, n is the principal quantum number (n=1 for ground state),
# m is the mass of the particle, and L is the width of the well. For an electron in a well, the effective mass m can be
# calculated as m = m_r * m_0, where m_r is the relative effective mass and m_0 is the free electron mass.
# The energy of a photon corresponding to this energy level can be related to its wavelength by the equation:
# E = h * c / λ, where c is the speed of light and λ is the wavelength.
# By equating the two expressions for energy, we can solve for the wavelength λ.

def ground_state_wavelength(L, mr):
    '''Given the width of an infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    # Constants
    h = 6.626e-34  # Planck constant in J*s
    c = 3e8        # Speed of light in m/s
    m_0 = 9.109e-31  # Free electron mass in kg

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Calculate the effective mass of the electron
    m = mr * m_0

    # Ground state energy (n=1) in the infinite square well
    E = (h**2) / (8 * m * L_m**2)

    # Wavelength of the photon corresponding to this energy
    lmbd = (h * c) / E

    # Convert the wavelength from meters to nanometers
    lmbd_nm = lmbd * 1e9

    return lmbd_nm



# Background: In mathematics, quadratic combinations of a set of numbers are expressions formed by combining these numbers
# using coefficients raised to the power of two. Specifically, given three numbers x, y, and z, a quadratic combination
# can be represented as i^2 * x + j^2 * y + k^2 * z, where i, j, k are non-negative integers. The task here is to generate
# the smallest N such combinations, ordered in ascending value, using the given three numbers. This involves systematically
# generating combinations, calculating their values, and sorting them to find the smallest N combinations.



def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    # Initialize a set to track unique quadratic combinations
    combinations_set = set()

    # Start iterating over possible i, j, k values
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Calculate the quadratic combination
                combination = i**2 * x + j**2 * y + k**2 * z
                # Add the combination to the set
                combinations_set.add(combination)

    # Convert the set to a sorted list
    sorted_combinations = sorted(combinations_set)

    # Take the smallest N combinations
    smallest_N_combinations = sorted_combinations[:N]

    # Convert to a numpy array before returning
    C = np.array(smallest_N_combinations)

    return C

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
