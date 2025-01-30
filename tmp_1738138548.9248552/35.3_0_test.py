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



def generate_quadratic_combinations(x, y, z, N):
    results = []
    for i in range(int(np.sqrt(N)) + 1):
        for j in range(int(np.sqrt(N)) + 1):
            for k in range(int(np.sqrt(N)) + 1):
                if len(results) < N:
                    results.append(i**2 * x + j**2 * y + k**2 * z)
                else:
                    max_val = max(results)
                    current = i**2 * x + j**2 * y + k**2 * z
                    if current < max_val:
                        results.remove(max_val)
                        results.append(current)
    return np.array(sorted(results))



# Background: In quantum mechanics, a quantum dot can be modeled as a particle in a 3D box (cuboid), where the energy levels are quantized. 
# The energy levels for a particle in a 3D box are given by the formula:
# E(n_x, n_y, n_z) = (h^2 / 8 * m * L_x^2) * n_x^2 + (h^2 / 8 * m * L_y^2) * n_y^2 + (h^2 / 8 * m * L_z^2) * n_z^2
# where n_x, n_y, n_z are quantum numbers (positive integers), L_x, L_y, L_z are the dimensions of the box, 
# m is the effective mass of the particle, and h is Planck's constant.
# The task is to calculate the smallest N non-zero energy levels for a cuboid quantum dot with given dimensions and effective mass.

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

    # Calculate the energy increment for each dimension
    E_a = (h**2) / (8 * m * a_m**2)
    E_b = (h**2) / (8 * m * b_m**2)
    E_c = (h**2) / (8 * m * c_m**2)

    # Generate all possible combinations of quantum numbers (n_x, n_y, n_z)
    # We start from 1 because n_x, n_y, n_z must be positive integers
    energy_levels = []
    for n_x, n_y, n_z in itertools.product(range(1, N+1), repeat=3):
        energy = n_x**2 * E_a + n_y**2 * E_b + n_z**2 * E_c
        energy_levels.append(energy)

    # Sort the energy levels and select the smallest N non-zero levels
    energy_levels = sorted(set(energy_levels))[:N]

    # Return the energy levels in descending order
    return np.array(sorted(energy_levels, reverse=True))


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