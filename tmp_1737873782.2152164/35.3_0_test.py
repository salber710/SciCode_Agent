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
    c_speed = 3e8  # Speed of light in m/s
    me = 9.109e-31  # Free electron mass in kg

    # Effective mass
    m = mr * me
    
    # Convert dimensions from nanometers to meters
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9
    
    # Initialize a set to hold unique energy levels
    energy_levels = set()
    
    # Iterate over possible values of i, j, k
    # We will search for combinations until we find at least N unique non-zero energy levels
    i_range = 10
    j_range = 10
    k_range = 10
    
    for i in range(1, i_range):
        for j in range(1, j_range):
            for k in range(1, k_range):
                # Calculate the energy increment
                E_ijk = (np.pi**2 * h**2 / (2 * m)) * (i**2 / a_m**2 + j**2 / b_m**2 + k**2 / c_m**2)
                energy_levels.add(E_ijk)
    
    # Convert to a sorted list and select the smallest N non-zero energy levels
    sorted_energy_levels = sorted(energy_levels, reverse=True)
    
    # Return the first N elements of the sorted list
    return np.array(sorted_energy_levels[:N])


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e