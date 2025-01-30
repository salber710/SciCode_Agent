import numpy as np



# Background: In optics, the propagation of light through layered media can be described using the transfer matrix method. 
# This method involves calculating the phase shift that light undergoes as it passes through each layer. 
# For a layer with refractive index n and thickness d, the phase shift φ is given by φ = (2 * π * n * d) / λ_in, 
# where λ_in is the wavelength of the incident light. In the case of a quarter-wavelength layer, the thickness d is 
# λ_b / (4 * n), where λ_b is the resonant wavelength. The transfer matrix for a single layer is a 2x2 matrix with 
# elements that depend on the phase shift. The matrix elements (A, B, C, D) are used to calculate the overall 
# transmission and reflection properties of the multilayer structure.


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
    # Calculate the phase shift for each layer
    phi1 = (2 * np.pi * n1 * (lambda_b / (4 * n1))) / lambda_in
    phi2 = (2 * np.pi * n2 * (lambda_b / (4 * n2))) / lambda_in
    
    # Calculate the matrix elements for each layer
    A1 = np.cos(phi1)
    B1 = 1j * np.sin(phi1) / n1
    C1 = 1j * n1 * np.sin(phi1)
    D1 = np.cos(phi1)
    
    A2 = np.cos(phi2)
    B2 = 1j * np.sin(phi2) / n2
    C2 = 1j * n2 * np.sin(phi2)
    D2 = np.cos(phi2)
    
    # Combine the matrices for the two layers
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    # Return the matrix as a tuple
    return np.array([[A, B], [C, D]])

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('39.1', 3)
target = targets[0]

assert np.allclose(matrix_elements(980, 980, 3.52, 2.95), target)
target = targets[1]

assert np.allclose(matrix_elements(1500, 980, 3.52, 2.95), target)
target = targets[2]

assert np.allclose(matrix_elements(800, 980, 3.52, 2.95), target)
