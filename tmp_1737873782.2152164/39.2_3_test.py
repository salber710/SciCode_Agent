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
    
    # Calculate (A + D) / 2
    avg_AD = (A + D) / 2

    # Calculate theta using the arccosine of the average
    theta = np.arccos(avg_AD)

    # If the real part of (A + D) / 2 is greater than 1, set the real part of theta to np.pi
    if avg_AD.real > 1:
        theta = np.pi + 0j

    return theta


try:
    targets = process_hdf5_to_tuple('39.2', 3)
    target = targets[0]
    assert np.allclose(get_theta(1+1j, 2-1j), target)

    target = targets[1]
    assert np.allclose(get_theta(1, 2j), target)

    target = targets[2]
    assert np.allclose(get_theta(-1, -1j), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e