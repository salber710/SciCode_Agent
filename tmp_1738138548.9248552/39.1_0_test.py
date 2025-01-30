from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: In optics, the phase shift of light as it passes through a medium is an important factor in determining how light interacts with materials. 
# When light travels through a layer of material with a refractive index, it experiences a phase shift that depends on the thickness of the layer and the wavelength of the light.
# For a quarter-wavelength layer, the thickness is set such that it is one-quarter of the wavelength of the light in the medium. 
# The phase shift φ for a layer of thickness d and refractive index n is given by φ = (2 * π * n * d) / λ_in, where λ_in is the wavelength of the incident light.
# The propagation matrix for a single layer can be represented as a 2x2 matrix with elements A, B, C, and D, which describe how the electric field of the light wave is transformed as it passes through the layer.
# For a quarter-wavelength layer, the matrix elements can be derived using the phase shift and are given by:
# A = D = cos(φ), B = j * sin(φ) / n, C = j * n * sin(φ), where j is the imaginary unit.


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
    # Calculate the thickness of each layer as a quarter-wavelength of lambda_b
    d1 = lambda_b / (4 * n1)
    d2 = lambda_b / (4 * n2)
    
    # Calculate the phase shifts for each layer
    phi1 = (2 * np.pi * n1 * d1) / lambda_in
    phi2 = (2 * np.pi * n2 * d2) / lambda_in
    
    # Calculate the matrix elements for the first layer
    A1 = np.cos(phi1)
    B1 = 1j * np.sin(phi1) / n1
    C1 = 1j * n1 * np.sin(phi1)
    D1 = np.cos(phi1)
    
    # Calculate the matrix elements for the second layer
    A2 = np.cos(phi2)
    B2 = 1j * np.sin(phi2) / n2
    C2 = 1j * n2 * np.sin(phi2)
    D2 = np.cos(phi2)
    
    # The overall matrix is the product of the two matrices
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    # Return the matrix as a tuple
    return np.array([[A, B], [C, D]])


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