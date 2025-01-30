from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools

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
    c = 3e8  # Speed of light in m/s
    me = 9.109e-31  # Free electron mass in kg
    
    # Convert L from nanometers to meters
    L_m = L * 1e-9
    
    # Calculate the ground state energy E1
    # E1 = (pi^2 * h^2) / (2 * m * L^2)
    # where m = mr * me
    m = mr * me
    E1 = (np.pi**2 * h**2) / (2 * m * L_m**2)
    
    # Calculate the corresponding photon wavelength lambda
    # E = hc / lambda => lambda = hc / E
    lmbd = (h * c) / E1
    
    # Convert lambda from meters to nanometers
    lmbd_nm = lmbd * 1e9
    
    return lmbd_nm





def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    
    # Initialize a set to hold unique quadratic combinations
    combinations = set()
    
    # Iterate over possible values of i, j, k
    # We will search for combinations until we find at least N unique values
    # Starting with a reasonable range for i, j, k to ensure we cover enough combinations
    max_range = 10  # This can be adjusted based on expected size of N
    while len(combinations) < N:
        for i in range(max_range):
            for j in range(max_range):
                for k in range(max_range):
                    # Calculate the quadratic combination
                    value = i**2 * x + j**2 * y + k**2 * z
                    combinations.add(value)
        
        # If not enough combinations are found, increase the range and try again
        max_range *= 2
    
    # Convert the set to a sorted list to get the smallest values
    sorted_combinations = sorted(combinations)
    
    # Return the first N elements of the sorted list
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