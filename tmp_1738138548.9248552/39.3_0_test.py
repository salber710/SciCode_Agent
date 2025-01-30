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



def get_theta(A, D):
    trace_half = (A + D) / 2
    if np.real(trace_half) > 1:
        theta = complex(np.pi, np.imag(np.arctanh(trace_half - 1)))
    else:
        theta = np.arccos(np.clip(trace_half, -1, 1))
    return theta



# Background: The reflection coefficient R for a distributed Bragg reflector (DBR) stack is determined by the 
# interaction of light with the periodic structure of alternating layers with different refractive indices. 
# The reflection is influenced by the phase shift and the number of layer pairs. The phase shift is represented 
# by the pseudo-angle θ, which is derived from the propagate matrix of the system. If θ is complex, the hyperbolic 
# sine function is used instead of the regular sine function to account for the imaginary component. The reflection 
# coefficient is calculated using the formula R = |(M[0,0] - M[1,1]) / (M[0,0] + M[1,1])|^2, where M is the 
# propagate matrix of the entire stack.


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
    
    # Calculate the matrix elements for a single pair
    A, B, C, D = matrix_elements(lambda_in, lambda_b, n1, n2)
    
    # Calculate the pseudo-angle theta
    theta = get_theta(A, D)
    
    # Calculate the reflection coefficient R
    if np.iscomplex(theta):
        # Use hyperbolic sine for complex theta
        sinh_theta = np.sinh(N * theta)
        R = np.abs((sinh_theta * B) / (sinh_theta * D))**2
    else:
        # Use sine for real theta
        sin_theta = np.sin(N * theta)
        R = np.abs((sin_theta * B) / (sin_theta * D))**2
    
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