import numpy as np



# Background: In optics, when light passes through a multi-layered medium, each layer causes a phase shift in the light wave.
# For a layer of thickness set as a quarter-wavelength, the phase shift φ can be calculated using φ = (2 * π * n * d) / λ_in,
# where d is the thickness of the layer, n is the refractive index of the layer, and λ_in is the wavelength of the incident light.
# For a quarter-wavelength layer, d = λ_b / (4 * n), where λ_b is the resonant wavelength.
# The propagation matrix for each layer in a two-layer system can be described by a 2x2 matrix with elements
# (A, B, C, D). For a single layer, the matrix is given by:
# M = [[cos(φ), (1j * sin(φ)) / η], [1j * η * sin(φ), cos(φ)]],
# where η is the optical admittance of the layer (η = n for non-magnetic materials).
# The overall propagation matrix for a system with multiple layers is obtained by multiplying the matrices of each layer.


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

    # Calculate the thickness of the layers (quarter-wavelength thickness)
    d1 = lambda_b / (4 * n1)
    d2 = lambda_b / (4 * n2)

    # Calculate the phase shifts for each layer
    phi1 = (2 * np.pi * n1 * d1) / lambda_in
    phi2 = (2 * np.pi * n2 * d2) / lambda_in

    # Calculate the propagation matrices for each layer
    M1 = np.array([[np.cos(phi1), 1j * np.sin(phi1) / n1],
                   [1j * n1 * np.sin(phi1), np.cos(phi1)]])

    M2 = np.array([[np.cos(phi2), 1j * np.sin(phi2) / n2],
                   [1j * n2 * np.sin(phi2), np.cos(phi2)]])

    # Overall propagation matrix is the product of the matrices for each layer
    M = np.dot(M1, M2)

    # Return the matrix as a tuple of its elements
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]

    return (A, B, C, D)

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('39.1', 3)
target = targets[0]

assert np.allclose(matrix_elements(980, 980, 3.52, 2.95), target)
target = targets[1]

assert np.allclose(matrix_elements(1500, 980, 3.52, 2.95), target)
target = targets[2]

assert np.allclose(matrix_elements(800, 980, 3.52, 2.95), target)
