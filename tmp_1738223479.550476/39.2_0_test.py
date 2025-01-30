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
    matrix (tuple of 4 complex numbers): Matrix used in the calculation of the transmission coefficient.
    '''

    # Calculate the angular frequency for the incident light
    frequency_in = 2 * np.pi / lambda_in

    # Calculate the thickness of each layer
    thickness1 = lambda_b / (4 * n1)
    thickness2 = lambda_b / (4 * n2)

    # Calculate the phase shifts for both layers
    phase_shift1 = frequency_in * n1 * thickness1
    phase_shift2 = frequency_in * n2 * thickness2

    # Use matrix representations with rotation matrices
    A1 = np.cos(phase_shift1) + 1j * np.sin(phase_shift1)
    B1 = -1j * np.sin(phase_shift1) / n1
    C1 = -1j * n1 * np.sin(phase_shift1)
    D1 = np.cos(phase_shift1) + 1j * np.sin(phase_shift1)

    A2 = np.cos(phase_shift2) + 1j * np.sin(phase_shift2)
    B2 = -1j * np.sin(phase_shift2) / n2
    C2 = -1j * n2 * np.sin(phase_shift2)
    D2 = np.cos(phase_shift2) + 1j * np.sin(phase_shift2)

    # Combine the matrices using a different combination method
    A = A1 * D2 - B1 * C2
    B = A1 * B2 - B1 * D2
    C = C1 * A2 - D1 * C2
    D = C1 * B2 - D1 * A2

    return (A, B, C, D)



# Background: In the context of optical physics, the pseudo-angle θ is used to describe the phase evolution
# of a wave as it propagates through a medium, such as a stack of dielectric layers. The propagate matrix
# M for multiple dielectric Bragg reflector (DBR) stacks is given by M = [[A, B], [C, D]], where A, B, C, and D 
# are complex numbers representing the combined effect of multiple layers on the incident light. The pseudo-angle θ 
# can be derived from the matrix elements. Specifically, θ is related to the trace of the matrix (A + D).
# If the real part of (A + D) / 2 exceeds 1, the real part of θ should be set to π, reflecting the physical constraint
# on the permissible values for angle θ in such optical systems.

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
    
    # Calculate the angle theta using the arccos function
    theta = np.arccos(trace_half)
    
    # If the real part of (A + D) / 2 is greater than 1, set the real part of theta to np.pi
    if np.real(trace_half) > 1:
        theta = np.pi + 0j  # Ensure theta remains a complex number with zero imaginary part

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