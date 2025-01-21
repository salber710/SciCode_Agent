import numpy as np



# Background: The phase shift $\phi$ of light passing through a medium is related to the optical path length,
# which is the product of the refractive index $n$ and the physical thickness of the layer $d$. For a quarter-wavelength layer,
# the thickness is $d = \lambda_b / (4n)$, where $\lambda_b$ is the resonant wavelength of the layer.
# The phase shift can be calculated as $\phi = (2 \pi n d) / \lambda_{in}$.
# The propagate matrix for a single layer can be described using the matrix elements A, B, C, D which relate to the
# wave propagation through the layer. For a quarter wavelength stack, assuming ideal conditions without absorption,
# the matrix elements can be derived from the phase shift: 
# A = D = cos(φ), B = 1j * sin(φ) / n, C = 1j * n * sin(φ).
# These matrices are used to calculate the transmission and reflection coefficients of multi-layered optical systems.


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
    
    # Calculate the phase shift φ for each layer
    phi1 = (2 * np.pi * n1 * (lambda_b / (4 * n1))) / lambda_in
    phi2 = (2 * np.pi * n2 * (lambda_b / (4 * n2))) / lambda_in
    
    # Calculate matrix elements for each layer
    A1 = D1 = np.cos(phi1)
    B1 = 1j * np.sin(phi1) / n1
    C1 = 1j * n1 * np.sin(phi1)
    
    A2 = D2 = np.cos(phi2)
    B2 = 1j * np.sin(phi2) / n2
    C2 = 1j * n2 * np.sin(phi2)
    
    # The overall matrix for the stack is the product of individual matrices
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    # Return as a numpy array
    matrix = np.array([[A, B], [C, D]])
    
    return matrix

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('39.1', 3)
target = targets[0]

assert np.allclose(matrix_elements(980, 980, 3.52, 2.95), target)
target = targets[1]

assert np.allclose(matrix_elements(1500, 980, 3.52, 2.95), target)
target = targets[2]

assert np.allclose(matrix_elements(800, 980, 3.52, 2.95), target)
