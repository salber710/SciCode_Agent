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
    matrix (2 by 2 numpy array containing 4 complex numbers): Matrix used in the calculation of the transmission coefficient.
    '''
    
    # Calculate the optical thickness for the quarter-wave layer
    d1 = lambda_b / (4 * n1)
    d2 = lambda_b / (4 * n2)

    # Calculate the phase shift for each layer
    phi1 = (2 * np.pi / lambda_in) * n1 * d1
    phi2 = (2 * np.pi / lambda_in) * n2 * d2

    # Calculate the propagation matrix elements
    A = np.cos(phi1) * np.cos(phi2) - (n1/n2) * np.sin(phi1) * np.sin(phi2)
    B = 1j * ((1/n2) * np.sin(phi2) * np.cos(phi1) + (1/n1) * np.sin(phi1) * np.cos(phi2))
    C = 1j * (n2 * np.sin(phi2) * np.cos(phi1) + n1 * np.sin(phi1) * np.cos(phi2))
    D = np.cos(phi1) * np.cos(phi2) - (n2/n1) * np.sin(phi1) * np.sin(phi2)

    return np.array([[A, B], [C, D]])


def get_theta(A, D):
    '''Calculates the angle theta used in the calculation of the transmission coefficient.
    If the value of (A + D) / 2 is greater than 1, keep the real part of theta as np.pi.
    Input:
    A (complex): Matrix factor from the calculation of phase shift and matrix factors.
    D (complex): Matrix factor from the calculation of phase shift and matrix factors.
    Output:
    theta (complex): Angle used in the calculation of the transmission coefficient.
    '''
    # Calculate the average of A and D
    avg_AD = (A + D) / 2

    # Calculate the real part of theta
    if np.real(avg_AD) > 1:
        real_theta = np.pi
    else:
        real_theta = np.arccos(np.real(avg_AD))

    # Calculate the imaginary part of theta
    imag_theta = np.arcsin(np.imag(avg_AD))

    # Combine the real and imaginary parts to form the complex theta
    theta = real_theta + 1j * imag_theta

    return theta




def R_coefficient(lambda_in, lambda_b, n1, n2, N):
    '''Calculates the total reflection coefficient for a given number of layer pairs.
    If theta is complex, uses hyperbolic sine functions in the calculation.
    Input:
    lambda_in (float): Wavelength of the incident light in nanometers.
    lambda_b (float): Resonant wavelength in nanometers.
    n1 (float): Refractive index of the first material.
    n2 (float): Refractive index of the second material.
    N (int): Number of pairs of layers.
    Output:
    R (float): Total reflection coefficient.
    '''
    
    # Step 1: Calculate the propagation matrix elements for one pair
    M = matrix_elements(lambda_in, lambda_b, n1, n2)
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]

    # Step 2: Calculate theta
    theta = get_theta(A, D)

    # Step 3: Calculate reflection coefficient R
    if np.iscomplex(theta):
        # Use hyperbolic sine for complex theta
        sinh_term = np.sinh(N * theta)
        cosh_term = np.cosh(N * theta)
        R = np.abs((sinh_term / cosh_term) ** 2)
    else:
        # Use sine for real theta
        sin_term = np.sin(N * theta)
        cos_term = np.cos(N * theta)
        R = np.abs((sin_term / cos_term) ** 2)
        
    return R


try:
    targets = process_hdf5_to_tuple('39.3', 4)
    target = targets[0]
    assert (np.isclose(R_coefficient(980, 980, 3.52, 2.95, 100),1,atol=10**-10)) == target

    target = targets[1]
    assert np.allclose(R_coefficient(1000, 980, 3.5, 3, 10), target)

    target = targets[2]
    assert np.allclose(R_coefficient(1500, 980, 3.52, 2.95, 20), target)

    target = targets[3]
    assert np.allclose(R_coefficient(800, 980, 3.52, 2.95, 20), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e