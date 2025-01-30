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
    
    # Calculate the phase shift by factoring in the wavelength ratio
    phase_factor = lambda_in / lambda_b

    # Calculate the effective optical path for each layer
    optical_path1 = n1 * (lambda_b / (4 * n1)) * phase_factor
    optical_path2 = n2 * (lambda_b / (4 * n2)) * phase_factor

    # Calculate the phase shift angles for each layer
    phi1 = 2 * np.pi * optical_path1 / lambda_in
    phi2 = 2 * np.pi * optical_path2 / lambda_in

    # Use trigonometric functions to compute matrix elements
    A1 = np.cos(phi1) - 1j * np.sin(phi1)
    B1 = 1j * np.tan(phi1) / n1
    C1 = 1j * n1 * np.tan(phi1)
    D1 = np.cos(phi1) - 1j * np.sin(phi1)

    A2 = np.cos(phi2) - 1j * np.sin(phi2)
    B2 = 1j * np.tan(phi2) / n2
    C2 = 1j * n2 * np.tan(phi2)
    D2 = np.cos(phi2) - 1j * np.sin(phi2)

    # Combine the matrices for the two layers using a different combination method
    A = A1 * A2 - B1 * C2
    B = A1 * B2 - B1 * D2
    C = C1 * A2 - D1 * C2
    D = C1 * B2 - D1 * D2

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