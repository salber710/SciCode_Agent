from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




def matrix_elements(lambda_in, lambda_b, n1, n2):
    '''Calculates the phase shift and the A/B/C/D matrix factors for a given wavelength.
    Input:
    lambda_in (float): Wavelength of the incident light in nanometers.
    lambda_b (float): Resonant wavelength in nanometers.
    n1 (float): Refractive index of the first material.
    n2 (float): Refractive index of the second material.
    Output:
    matrix (2 by 2 numpy array containing 4 complex numbers): Matrix used in the calculation of the transmission coefficient.
    '''
    
    # Calculate the phase shift for each layer, assuming quarter-wavelength thickness
    # Phase shift φ = (2 * π * n * thickness) / λ_in
    # For quarter-wavelength thickness at λ_b, thickness = λ_b / (4 * n)
    phi1 = (2 * np.pi * n1 * lambda_b / (4 * n1)) / lambda_in
    phi2 = (2 * np.pi * n2 * lambda_b / (4 * n2)) / lambda_in
    
    # Calculate the elements of the propagation matrix for each layer
    # Using the formulas:
    # For a single layer with phase shift φ:
    # A = D = cos(φ)
    # B = j * sin(φ) / n
    # C = j * n * sin(φ)
    
    A1 = np.cos(phi1)
    B1 = 1j * np.sin(phi1) / n1
    C1 = 1j * n1 * np.sin(phi1)
    D1 = A1
    
    A2 = np.cos(phi2)
    B2 = 1j * np.sin(phi2) / n2
    C2 = 1j * n2 * np.sin(phi2)
    D2 = A2
    
    # Combine the two matrices into one using matrix multiplication
    # [A, B; C, D] = [A1, B1; C1, D1] * [A2, B2; C2, D2]
    
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    # Return the combined matrix elements as a tuple
    matrix = (A, B, C, D)

    return matrix


try:
    targets = process_hdf5_to_tuple('39.1', 3)
    target = targets[0]
    assert np.allclose(matrix_elements(980, 980, 3.52, 2.95), target)

    target = targets[1]
    assert np.allclose(matrix_elements(1500, 980, 3.52, 2.95), target)

    target = targets[2]
    assert np.allclose(matrix_elements(800, 980, 3.52, 2.95), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e