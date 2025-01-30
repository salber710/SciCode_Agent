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
    matrix (tuple of 4 complex numbers): Matrix elements (A, B, C, D) used in the calculation of the transmission coefficient.
    '''
    
    # Calculate the phase shift for each layer
    phi1 = (2 * np.pi * n1 * lambda_b / lambda_in) / 4  # Quarter-wavelength for layer 1
    phi2 = (2 * np.pi * n2 * lambda_b / lambda_in) / 4  # Quarter-wavelength for layer 2
    
    # Calculate the matrix elements
    A = np.cos(phi1) * np.cos(phi2) - (n2 / n1) * np.sin(phi1) * np.sin(phi2)
    B = (1j / n1) * (np.sin(phi1) * np.cos(phi2) + (n2 / n1) * np.cos(phi1) * np.sin(phi2))
    C = (1j * n1) * (np.cos(phi1) * np.sin(phi2) + (n1 / n2) * np.sin(phi1) * np.cos(phi2))
    D = np.cos(phi1) * np.cos(phi2) - (n1 / n2) * np.sin(phi1) * np.sin(phi2)
    
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