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
    
    # Define the angular frequency omega for the incident wavelength
    omega = 2 * np.pi / lambda_in

    # Calculate the optical thickness for each layer
    optical_thickness1 = omega * lambda_b / (4 * lambda_in)
    optical_thickness2 = omega * lambda_b / (4 * lambda_in)
    
    # Calculate the phase shifts using the optical thickness
    phi1 = n1 * optical_thickness1
    phi2 = n2 * optical_thickness2
    
    # Calculate the matrix elements for the first layer using trigonometric identities
    A1 = np.exp(-1j * phi1)
    B1 = np.exp(1j * phi1) / (1j * n1 * omega)
    C1 = 1j * n1 * omega * np.exp(-1j * phi1)
    D1 = np.exp(1j * phi1)
    
    # Calculate the matrix elements for the second layer
    A2 = np.exp(-1j * phi2)
    B2 = np.exp(1j * phi2) / (1j * n2 * omega)
    C2 = 1j * n2 * omega * np.exp(-1j * phi2)
    D2 = np.exp(1j * phi2)
    
    # Compute the combined matrix elements using the product formula
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