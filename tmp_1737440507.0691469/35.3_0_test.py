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


# Background: To generate the smallest quadratic combinations of three numbers x, y, z,
# we consider expressions of the form i^2 * x + j^2 * y + k^2 * z, where i, j, k are non-negative integers.
# The task is to find the smallest N such unique combinations.
# This requires iterating over possible values of i, j, and k, computing the quadratic combination,
# and collecting them in a set to ensure uniqueness. Once we have enough unique combinations,
# we sort them and select the smallest N values.



def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    results = set()
    
    # We use a large enough range to ensure we can find at least N unique combinations
    # Start iterating over possible i, j, k values
    for i in range(int(N**0.5) + 1):  # Rough estimate to limit i, j, k
        for j in range(int(N**0.5) + 1):
            for k in range(int(N**0.5) + 1):
                # Calculate the quadratic combination
                combination = i**2 * x + j**2 * y + k**2 * z
                # Add the combination to the set
                results.add(combination)
                # If we have enough unique results, break
                if len(results) >= N:
                    break
            if len(results) >= N:
                break
        if len(results) >= N:
            break

    # Convert the set to a sorted list
    sorted_results = sorted(results)

    # Return the first N elements as a numpy array
    return np.array(sorted_results[:N])



# Background: In a 3D quantum dot, modeled as a cuboid, the energy levels for an electron can be calculated
# as the sum of the quantized energy levels in each dimension. The energy for each dimension is given by the
# formula E_n = (h^2 * n^2) / (8 * m * L^2), where h is the Planck constant, n is the quantum number (n=1, 2, 3...),
# m is the effective mass of the electron, and L is the dimension size (a, b, or c in this case). 
# To find the smallest N non-zero energy levels, we combine these energies from each dimension using combinations
# of quantum numbers (i, j, k) for the dimensions a, b, and c respectively. We calculate these energies, 
# sort them, and then select the smallest N. The energies are then converted to photon wavelengths using the 
# relation E = h * c / λ, where c is the speed of light and λ is the wavelength.



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
    c = 3e8        # Speed of light in m/s
    m_0 = 9.109e-31  # Free electron mass in kg

    # Convert dimensions from nanometers to meters
    a_m = a * 1e-9
    b_m = b * 1e-9
    c_m = c * 1e-9

    # Calculate the effective mass of the electron
    m = mr * m_0

    # To store energy levels
    energy_levels = set()

    # Rough estimate: Iterate over possible quantum numbers i, j, k
    # The range is arbitrary but should be large enough to ensure we get enough non-zero energy levels
    for i in range(1, int(N**0.5) + 2):
        for j in range(1, int(N**0.5) + 2):
            for k in range(1, int(N**0.5) + 2):
                # Calculate energy for the combination of quantum numbers i, j, k
                E_i = (h**2 * i**2) / (8 * m * a_m**2)
                E_j = (h**2 * j**2) / (8 * m * b_m**2)
                E_k = (h**2 * k**2) / (8 * m * c_m**2)
                # Total energy for this combination
                E_total = E_i + E_j + E_k
                # Add to set to ensure uniqueness
                energy_levels.add(E_total)

    # Convert energy levels set to a sorted list
    sorted_energy_levels = sorted(energy_levels)

    # Select the smallest N non-zero energies
    smallest_energies = sorted_energy_levels[:N]

    # Convert energies to wavelengths
    wavelengths = [(h * c) / E for E in smallest_energies]

    # Return wavelengths in descending order
    return np.array(sorted(wavelengths, reverse=True))

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
