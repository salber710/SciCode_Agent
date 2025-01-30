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
    matrix (tuple of 4 complex numbers): Matrix used in the calculation of the transmission coefficient.
    '''
    
    # Calculate the phase shift using an alternative method with normalized thickness
    normalized_thickness1 = lambda_b / (4 * lambda_in)
    normalized_thickness2 = lambda_b / (4 * lambda_in)

    # Compute phase shifts based on normalized thickness
    phi1 = 2 * np.pi * n1 * normalized_thickness1
    phi2 = 2 * np.pi * n2 * normalized_thickness2

    # Calculate matrix elements using polar coordinates to enhance numerical stability
    A1 = np.exp(1j * phi1) * np.cos(phi1)
    B1 = (np.exp(-1j * phi1) - 1) / n1
    C1 = n1 * (np.exp(-1j * phi1) - 1)
    D1 = np.exp(1j * phi1) * np.cos(phi1)

    A2 = np.exp(1j * phi2) * np.cos(phi2)
    B2 = (np.exp(-1j * phi2) - 1) / n2
    C2 = n2 * (np.exp(-1j * phi2) - 1)
    D2 = np.exp(1j * phi2) * np.cos(phi2)

    # Combine matrices using a cross-multiplication technique
    A = A1 * D2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * D2 + D1 * C2
    D = C1 * B2 + D1 * D2

    return (A, B, C, D)


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