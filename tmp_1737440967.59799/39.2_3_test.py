import numpy as np

# Background: In optics, the phase shift of light as it propagates through a medium is a crucial factor in understanding interference and transmission properties. 
# The phase shift (ϕ) is given by the formula ϕ = (2 * π * n * d) / λ_in, where n is the refractive index, d is the thickness of the layer, and λ_in is the wavelength of the incident light.
# For a quarter-wavelength layer, the thickness d is set to λ_b / (4 * n), where λ_b is the resonant wavelength.
# The propagation matrix for a single layer can be described by the transfer matrix method, which is a 2x2 matrix with elements A, B, C, and D.
# For a quarter-wavelength layer, the matrix elements are typically: A = D = cos(ϕ), B = i * sin(ϕ) / (n * k), C = i * n * k * sin(ϕ), where k is the wave number (k = 2 * π / λ_b).
# These matrices are used in optics to calculate the transmission and reflection coefficients of multi-layered structures.


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

    # Calculate the wave number for the resonant wavelength
    k = 2 * np.pi / lambda_b

    # Calculate the phase shift for each layer
    # Layer 1
    d1 = lambda_b / (4 * n1)
    phi1 = (2 * np.pi * n1 * d1) / lambda_in
    
    # Layer 2
    d2 = lambda_b / (4 * n2)
    phi2 = (2 * np.pi * n2 * d2) / lambda_in

    # Calculate matrix elements for the first layer
    A1 = np.cos(phi1)
    B1 = 1j * np.sin(phi1) / (n1 * k)
    C1 = 1j * n1 * k * np.sin(phi1)
    D1 = np.cos(phi1)

    # Calculate matrix elements for the second layer
    A2 = np.cos(phi2)
    B2 = 1j * np.sin(phi2) / (n2 * k)
    C2 = 1j * n2 * k * np.sin(phi2)
    D2 = np.cos(phi2)

    # Combine the matrices of the two layers to get the total matrix
    # M_total = M1 * M2
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2

    matrix = np.array([[A, B],
                       [C, D]], dtype=complex)

    return matrix



# Background: In the context of optics and particularly in the analysis of multilayer structures like distributed Bragg reflectors (DBRs), the concept of the pseudo-angle θ comes into play when dealing with the propagation matrix. The propagation matrix for a multilayer stack is characterized by elements A, B, C, and D. The pseudo-angle θ is derived from the trace of the matrix, which is the sum of its diagonal elements (A + D). This trace relates to the angle through the relation cos(θ) = (A + D) / 2. If the trace divided by 2 exceeds the value of 1, which is outside the domain of the arccosine function, this indicates a potential issue in the calculation, often due to numerical errors or specific physical conditions. In such cases, the real part of θ is set to π, ensuring that it remains within a valid physical range.

def get_theta(A, D):
    '''Calculates the angle theta used in the calculation of the transmission coefficient.
    If the value of (A + D) / 2 is greater than 1, keep the real part of theta as np.pi.
    Input:
    A (complex): Matrix factor from the calculation of phase shift and matrix factors.
    D (complex): Matrix factor from the calculation of phase shift and matrix factors.
    Output:
    theta (complex): Angle used in the calculation of the transmission coefficient.
    '''


    # Calculate (A + D) / 2
    trace_half = (A + D) / 2

    # Calculate theta
    if trace_half.real > 1:
        # If the real part of trace_half is greater than 1, set real part of theta to pi
        theta_real = np.pi
    else:
        # Otherwise, calculate the arccos of the real part of trace_half
        theta_real = np.arccos(trace_half.real)

    # The imaginary part of theta is simply the imaginary part of the arcosine
    theta_imag = np.arccos(trace_half).imag
    
    # Combine them into a complex number
    theta = theta_real + 1j * theta_imag

    return theta

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('39.2', 3)
target = targets[0]

assert np.allclose(get_theta(1+1j, 2-1j), target)
target = targets[1]

assert np.allclose(get_theta(1, 2j), target)
target = targets[2]

assert np.allclose(get_theta(-1, -1j), target)
