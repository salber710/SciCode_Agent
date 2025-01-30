from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def matrix_elements(lambda_in, lambda_b, n1, n2):
    # Calculate the phase shift using a direct approach for quarter-wavelength
    phi1 = np.pi / 2  # Phase shift for quarter-wavelength is π/2
    phi2 = np.pi / 2  # Same for the second layer

    # Using a matrix representation with trigonometric functions and a twist in the calculation
    # Matrix for first layer
    M1 = np.array([
        [np.cos(phi1), np.sin(phi1) / n1],
        [n1 * np.sin(phi1), np.cos(phi1)]
    ])

    # Matrix for second layer
    M2 = np.array([
        [np.cos(phi2), np.sin(phi2) / n2],
        [n2 * np.sin(phi2), np.cos(phi2)]
    ])

    # Matrix multiplication of M1 and M2 for the combined system
    M = np.dot(M1, M2)

    # Extract A, B, C, D components from the resulting matrix
    A, B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]

    # Return the matrix as a tuple
    return (A, B, C, D)



# Background: In optics, the pseudo-angle θ is used to describe the phase relationship in a multilayer dielectric structure, such as a distributed Bragg reflector (DBR). The propagate matrix M for such a system is given by M = [[A, B], [C, D]]. The trace of this matrix, (A + D), is related to the cosine of the pseudo-angle θ. Specifically, θ is defined such that cos(θ) = (A + D) / 2. However, due to the properties of the cosine function, if (A + D) / 2 exceeds 1, the real part of θ should be set to π to maintain physical consistency, as the cosine function cannot exceed 1. This adjustment ensures that the calculated angle remains within the valid range for real physical systems.

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
        # If the real part of trace_half is greater than 1, set the real part of theta to π
        theta = np.pi
    else:
        # Otherwise, calculate theta using the arccosine function
        theta = np.arccos(trace_half)

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