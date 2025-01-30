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
    
    # Return the matrix as a numpy array
    return np.array([[A, B], [C, D]])



# Background: In the context of optical multilayer structures, the pseudo-angle θ is used to describe the 
# phase relationship between the electric fields at the boundaries of the layers. The matrix M, which is 
# derived from the transfer matrix method, has elements A, B, C, and D. The trace of this matrix, (A + D), 
# is related to the cosine of the pseudo-angle θ. Specifically, θ is defined such that cos(θ) = (A + D) / 2. 
# However, due to the properties of the cosine function, if (A + D) / 2 exceeds 1, the real part of θ is 
# set to π to ensure a valid angle within the principal range of the arccosine function. This adjustment 
# is necessary because the arccosine function is only defined for inputs in the range [-1, 1].

def get_theta(A, D):
    '''Calculates the angle theta used in the calculation of the transmission coefficient.
    If the value of (A + D) / 2 is greater than 1, keep the real part of theta as np.pi.
    Input:
    A (complex): Matrix factor from the calculation of phase shift and matrix factors.
    D (complex): Matrix factor from the calculation of phase shift and matrix factors.
    Output:
    theta (complex): Angle used in the calculation of the transmission coefficient.
    '''


    # Calculate the cosine of the pseudo-angle theta
    cos_theta = (A + D) / 2

    # Determine the real part of theta
    if np.real(cos_theta) > 1:
        real_theta = np.pi
    else:
        real_theta = np.arccos(np.real(cos_theta))

    # Calculate the imaginary part of theta
    imag_theta = np.arccosh(cos_theta).imag

    # Combine the real and imaginary parts to form the complex theta
    theta = real_theta + 1j * imag_theta

    return theta

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('39.2', 3)
target = targets[0]

assert np.allclose(get_theta(1+1j, 2-1j), target)
target = targets[1]

assert np.allclose(get_theta(1, 2j), target)
target = targets[2]

assert np.allclose(get_theta(-1, -1j), target)
