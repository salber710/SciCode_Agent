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

    # Calculate the reduced wavenumber for the incident light
    reduced_k = 2 * np.pi / lambda_in

    # Set the quarter-wavelength thickness for each layer
    thickness1 = lambda_b / (4 * n1)
    thickness2 = lambda_b / (4 * n2)

    # Determine the phase shifts for the layers
    phi1 = (2 * np.pi * n1 * thickness1) / lambda_in
    phi2 = (2 * np.pi * n2 * thickness2) / lambda_in

    # Use hyperbolic identities to compute the matrix elements
    A1 = np.exp(phi1)
    B1 = -np.sinh(phi1) / (n1 * reduced_k)
    C1 = -n1 * reduced_k * np.sinh(phi1)
    D1 = np.exp(phi1)

    A2 = np.exp(phi2)
    B2 = -np.sinh(phi2) / (n2 * reduced_k)
    C2 = -n2 * reduced_k * np.sinh(phi2)
    D2 = np.exp(phi2)

    # Calculate the combined matrix elements using a distinct order of multiplication
    A = A1 * D2 + C1 * B2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = B1 * C2 + D1 * A2

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