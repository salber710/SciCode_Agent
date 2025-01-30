from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools

def ground_state_wavelength(L, mr):
    # Constants
    h = 6.62607015e-34  # Planck constant in J*s, using more precise value
    c = 299792458       # Speed of light in m/s, using exact value
    m_e = 9.10938356e-31  # Free electron mass in kg, using more precise value

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Calculate the effective mass
    m = mr * m_e

    # Ground state energy using n=1
    E = (h**2) / (8 * m * L_m**2)

    # Wavelength of the photon corresponding to this energy
    lambda_m = h * c / E

    # Convert wavelength back to nanometers
    lambda_nm = lambda_m * 1e9

    return lambda_nm



# Background: In mathematics, a quadratic combination of numbers involves expressions of the form i^2x + j^2y + k^2z, 
# where i, j, and k are integers. The task is to generate the smallest N such combinations using the given numbers x, y, and z.
# The coefficients i, j, and k start from 0 and can increase to generate different combinations. The goal is to find the 
# smallest N values of these combinations and return them in ascending order. This involves iterating over possible values 
# of i, j, and k, calculating the quadratic combination, and storing the smallest results.



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
    
    # We will iterate over a range of values for i, j, k to generate combinations
    # The range is chosen to ensure we have enough combinations to select the smallest N
    # This is a heuristic choice; adjust as needed for larger N
    max_range = int(np.sqrt(N)) + 1
    
    for i, j, k in itertools.product(range(max_range), repeat=3):
        # Calculate the quadratic combination
        combination = i**2 * x + j**2 * y + k**2 * z
        # Add the combination to the set
        combinations.add(combination)
    
    # Convert the set to a sorted list
    sorted_combinations = sorted(combinations)
    
    # Return the first N elements as a numpy array
    return np.array(sorted_combinations[:N])


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e