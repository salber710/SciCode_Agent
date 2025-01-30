from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools

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
    c = 3e8  # Speed of light in m/s
    m_free = 9.109e-31  # Free electron mass in kg

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Effective mass calculation
    m_eff = mr * m_free

    # Ground state energy: E = (h^2) / (8 * m_eff * L^2)
    E = h**2 / (8 * m_eff * L_m**2)

    # Calculate the wavelength in meters
    lmbd_m = (h * c) / E

    # Convert the wavelength to nanometers using a distinct method
    lmbd = lmbd_m / (1e-9)

    return lmbd




def generate_quadratic_combinations(x, y, z, N):
    '''With three numbers given, return an array with the size N that contains the smallest N numbers which are quadratic combinations of the input numbers.
    Input:
    x (float): The first number.
    y (float): The second number.
    z (float): The third number.
    Output:
    C (size N numpy array): The collection of the quadratic combinations.
    '''
    def next_combinations(i, j, k):
        """Generate the next combinations by incrementing indices."""
        yield (i + 1, j, k)
        yield (i, j + 1, k)
        yield (i, j, k + 1)

    # A priority queue to store the combinations
    min_heap = [(x + y + z, 1, 1, 1)]
    visited = set((1, 1, 1))
    results = []

    while len(results) < N:
        value, i, j, k = heapq.heappop(min_heap)
        results.append(value)

        for ni, nj, nk in next_combinations(i, j, k):
            if (ni, nj, nk) not in visited:
                visited.add((ni, nj, nk))
                new_value = ni**2 * x + nj**2 * y + nk**2 * z
                heapq.heappush(min_heap, (new_value, ni, nj, nk))
    
    return np.array(results)




def absorption(mr, a, b, c, N):
    # Constants
    h = 6.626e-34  # Planck constant in J*s
    m_free = 9.109e-31  # Free electron mass in kg

    # Convert dimensions from nanometers to meters
    a_m, b_m, c_m = a * 1e-9, b * 1e-9, c * 1e-9

    # Effective mass calculation
    m_eff = mr * m_free

    # List to store energy levels
    energy_levels = []

    # Quantum numbers i, j, k start from 1 to avoid zero energy states
    # Instead of itertools.product, use nested loops with a different range for each dimension
    max_quantum_number = int(np.ceil((N**(1/3)) * 2))  # Estimation for loop limits

    for i in range(1, max_quantum_number):
        for j in range(1, max_quantum_number):
            for k in range(1, max_quantum_number):
                energy = (h**2 / (8 * m_eff)) * ((i**2 / a_m**2) + (j**2 / b_m**2) + (k**2 / c_m**2))
                energy_levels.append(energy)

    # Sort and filter unique energy levels
    energy_levels = sorted(set(energy_levels))

    # Return the largest N non-zero energies in descending order
    return np.array(energy_levels[-N:][::-1])


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