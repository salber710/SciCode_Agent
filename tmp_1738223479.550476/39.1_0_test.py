from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: In optics, when dealing with thin film interference, the concept of a quarter-wavelength layer refers to a layer
# whose thickness is one quarter of the wavelength of light for which the layer is designed (the resonant wavelength, λ_b).
# This design creates constructive or destructive interference effects. The phase shift φ for a layer of refractive index n and 
# thickness d is given by φ = (2 * π * n * d) / λ_in, where λ_in is the wavelength of the incident light. For a quarter-wavelength
# layer, d = λ_b / (4 * n). The propagation matrix for such a layer is represented in terms of its matrix elements A, B, C, and D.
# The matrix elements for a single layer can be calculated using the phase shift: 
# A = D = cos(φ), B = j * sin(φ) / (n * k), C = j * n * k * sin(φ), where k = 2π/λ_in and j is the imaginary unit.


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
    
    # Calculate the wavenumber k for the incident wavelength
    k = 2 * np.pi / lambda_in
    
    # Calculate the thickness of each layer for quarter-wavelength condition
    d1 = lambda_b / (4 * n1)
    d2 = lambda_b / (4 * n2)
    
    # Calculate the phase shifts for both layers
    phi1 = (2 * np.pi * n1 * d1) / lambda_in
    phi2 = (2 * np.pi * n2 * d2) / lambda_in
    
    # Calculate the matrix elements for the first layer
    A1 = np.cos(phi1)
    B1 = 1j * np.sin(phi1) / (n1 * k)
    C1 = 1j * n1 * k * np.sin(phi1)
    D1 = np.cos(phi1)
    
    # Calculate the matrix elements for the second layer
    A2 = np.cos(phi2)
    B2 = 1j * np.sin(phi2) / (n2 * k)
    C2 = 1j * n2 * k * np.sin(phi2)
    D2 = np.cos(phi2)
    
    # Combine the matrices for the two layers
    # The resulting matrix is the product of the two matrices: M = M1 * M2
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    matrix = np.array([[A, B], [C, D]], dtype=complex)
    
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