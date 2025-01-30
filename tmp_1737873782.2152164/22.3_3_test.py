from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy



def Rlnm(l, n, m, k, z, N_t):
    '''Function to calculate the translation coefficient (R|R)lmn.
    Input
    l : int
        The principal quantum number for the expansion.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    k : float
        Wavevector of the optical beam.
    z : float
        The translation distance along z direction
    N_t : int
        Truncated space size.
    Output
    (R|R)lmn : complex
        The translation coefficient (R|R)lmn.
    '''

    # Initialize the translation coefficient
    translation_coefficient = 0.0

    # Precompute common factors
    kz = k * z

    # Recursively calculate the translation coefficient
    for l_prime in range(0, N_t):
        for s in range(-l_prime, l_prime + 1):
            # Calculate the spherical Bessel functions
            jn_kr = spherical_jn(l_prime, kz)
            yn_kr = spherical_yn(l_prime, kz)

            # Compute the translation coefficient using the recursion relation
            # Here, we use the recursion for spherical Bessel functions
            # R_l^m(r_q) is expanded using spherical harmonics and Bessel functions
            # This is a simplified approach and may require additional terms depending on the complexity
            # of the problem and the specific recursion involved.
            term = ((2 * l_prime + 1) * (1j)**(l - l_prime) *
                    (jn_kr + 1j * yn_kr)) 

            # Accumulate the contribution for this term
            translation_coefficient += term

    return translation_coefficient


def Tnvm(n, v, m, Q):
    '''Function to calculate the rotation coefficient Tnvm.
    Input
    n : int
        The principal quantum number for both expansion and reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    v : int
        The magnetic quantum number for the expansion.
    Q : matrix of shape(3, 3)
        The rotation matrix.
    Output
    T : complex
        The rotation coefficient Tnvm.
    '''

    # Import necessary functions from scipy


    # Initialize the rotation coefficient
    T = 0.0

    # Iterate over possible values of the magnetic quantum number
    for mu in range(-n, n + 1):
        # Compute the Wigner D-matrix element
        D = 0.0
        for i in range(3):
            for j in range(3):
                D += Q[i, j] * sph_harm(mu, n, i, j).real  # Assuming a real rotation matrix

        # Accumulate the contribution for this term
        T += D * sph_harm(mu, n, 0, 0).real  # Using spherical harmonics

    return T



def compute_BRnm(r0, B, n, m, wl, N_t):
    '''Function to calculate the reexpansion coefficient BR_nm.
    Input
    r_0 : array
        Translation vector.
    B : matrix of shape(N_t + 1, 2 * N_t + 1)
        Expansion coefficients of the elementary regular solutions.
    n : int
        The principal quantum number for the reexpansion.
    m : int
        The magnetic quantum number for the reexpansion.
    wl : float
        Wavelength of the optical beam.
    N_t : int
        Truncated space size.
    Output
    BR_nm : complex
        Reexpansion coefficient BR_nm of the elementary regular solutions.
    '''
    
    # Calculate the magnitude of the translation vector
    r0_magnitude = np.linalg.norm(r0)
    
    # Calculate the direction cosines for the rotation matrix
    if r0_magnitude != 0:
        direction_cosines = r0 / r0_magnitude
    else:
        direction_cosines = np.array([0, 0, 1])  # Default to z-direction if zero vector
    
    # Define the initial and final unit vectors
    z_hat = np.array([0, 0, 1])
    
    # Calculate the rotation matrix to align z-axis with r0 direction using Rodrigues' rotation formula
    v = np.cross(z_hat, direction_cosines)
    s = np.linalg.norm(v)
    c = np.dot(z_hat, direction_cosines)
    
    if s == 0:  # No rotation needed if r0 is aligned with z_hat
        Q = np.eye(3)
    else:
        k = v / s
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        Q = np.eye(3) + K + np.dot(K, K) * ((1 - c) / (s**2))

    # Calculate the translation coefficient using Rlnm function
    # Assuming the Rlnm function is defined to compute the translation coefficient
    k = 2 * np.pi / wl  # Wave number
    Rlnm_coeff = Rlnm(n, n, m, k, r0_magnitude, N_t)
    
    # Calculate the rotation coefficient using Tnvm function
    # Assuming the Tnvm function is defined to compute the rotation coefficient
    Tnvm_coeff = Tnvm(n, m, m, Q)
    
    # Compute the reexpansion coefficient
    BR_nm = B[n, m + N_t] * Rlnm_coeff * Tnvm_coeff
    
    return BR_nm


try:
    targets = process_hdf5_to_tuple('22.3', 4)
    target = targets[0]
    r0 = np.array([0.5, 0, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t] = 1
    wl = 2 * np.pi
    n = 2
    m = 1
    assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)

    target = targets[1]
    r0 = np.array([0.5, 0.5, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t] = 1
    wl = 2 * np.pi
    n = 2
    m = 1
    assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)

    target = targets[2]
    r0 = np.array([0.5, 1, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t] = 1
    wl = 2 * np.pi
    n = 2
    m = 1
    assert np.allclose(compute_BRnm(r0, B, n, m, wl, N_t), target)

    target = targets[3]
    r0 = np.array([0, 0.5, 0])
    N_t = 5
    B = np.zeros((N_t + 1, 2 * N_t + 1))
    B[1, N_t + 1] = 1
    wl = 2 * np.pi
    n = 2
    m = 2
    assert (compute_BRnm(r0, B, n, m, wl, N_t) == 0) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e