import numpy as np

# Background: In optics, the propagation of light through layered media can be described using the transfer matrix method.
# This method involves calculating the phase shift and the propagation matrix for each layer. The phase shift, φ, is given by
# φ = (2 * π * n * d) / λ_in, where n is the refractive index of the layer, d is the thickness of the layer, and λ_in is the
# wavelength of the incident light. For a quarter-wavelength layer, d = λ_b / (4 * n), where λ_b is the resonant wavelength.
# The propagation matrix for a single layer is given by:
# | A B |
# | C D |
# where A = D = cos(φ), B = (i/n) * sin(φ), and C = i * n * sin(φ). The overall matrix for a two-layer system is the product
# of the matrices for each layer.


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
    B1 = (1j / n1) * np.sin(phi1)
    C1 = 1j * n1 * np.sin(phi1)
    D1 = np.cos(phi1)
    
    # Calculate the matrix elements for the second layer
    A2 = np.cos(phi2)
    B2 = (1j / n2) * np.sin(phi2)
    C2 = 1j * n2 * np.sin(phi2)
    D2 = np.cos(phi2)
    
    # Calculate the overall matrix by multiplying the matrices of the two layers
    A = A1 * A2 + B1 * C2
    B = A1 * B2 + B1 * D2
    C = C1 * A2 + D1 * C2
    D = C1 * B2 + D1 * D2
    
    # Return the matrix as a tuple
    return (A, B, C, D)



# Background: In the context of optics and the transfer matrix method, the pseudo-angle θ is used to describe the 
# phase relationship in a multilayer dielectric structure, such as a distributed Bragg reflector (DBR). The matrix 
# M = [[A, B], [C, D]] represents the combined effect of multiple layers. The pseudo-angle θ is derived from the 
# trace of this matrix, specifically from the expression (A + D) / 2. This value is related to the cosine of the 
# angle θ. If (A + D) / 2 is greater than 1, it indicates that the angle is complex, and the real part of θ should 
# be set to π to maintain physical consistency, as the cosine of an angle cannot exceed 1 in real numbers.

def get_theta(A, D):
    '''Calculates the angle theta used in the calculation of the transmission coefficient.
    If the value of (A + D) / 2 is greater than 1, keep the real part of theta as np.pi.
    Input:
    A (complex): Matrix factor from the calculation of phase shift and matrix factors.
    D (complex): Matrix factor from the calculation of phase shift and matrix factors.
    Output:
    theta (complex): Angle used in the calculation of the transmission coefficient.
    '''

    
    # Calculate the trace of the matrix divided by 2
    trace_half = (A + D) / 2
    
    # Calculate the angle theta
    if np.real(trace_half) > 1:
        # If the real part of trace_half is greater than 1, set the real part of theta to pi
        theta = np.pi + 1j * np.arccosh(trace_half)
    else:
        # Otherwise, calculate theta using arccos
        theta = np.arccos(trace_half)
    
    return theta


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('39.2', 3)
target = targets[0]

assert np.allclose(get_theta(1+1j, 2-1j), target)
target = targets[1]

assert np.allclose(get_theta(1, 2j), target)
target = targets[2]

assert np.allclose(get_theta(-1, -1j), target)
