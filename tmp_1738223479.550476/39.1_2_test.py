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
    
    # Calculate the angular frequency omega for the resonant wavelength
    omega_b = 2 * np.pi / lambda_b

    # Calculate the thickness for each layer using the quarter-wavelength condition
    thickness1 = lambda_b / (4 * n1)
    thickness2 = lambda_b / (4 * n2)

    # Calculate the phase shifts for both layers using the incident wavelength
    phi1 = omega_b * n1 * thickness1 / lambda_in
    phi2 = omega_b * n2 * thickness2 / lambda_in

    # For a single layer, the phase shift matrix is represented using hyperbolic functions
    A1 = np.cosh(phi1)
    B1 = -1j * np.sinh(phi1) / n1
    C1 = -1j * n1 * np.sinh(phi1)
    D1 = np.cosh(phi1)

    A2 = np.cosh(phi2)
    B2 = -1j * np.sinh(phi2) / n2
    C2 = -1j * n2 * np.sinh(phi2)
    D2 = np.cosh(phi2)

    # Compute the combined matrix elements using matrix multiplication for the two layers
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
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