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
        The translation distance along z direction.
    N_t : int
        Truncated space size.
    Output
    (R|R)lmn : complex
        The translation coefficient (R|R)lmn.
    '''



    # Calculate r0 as the Euclidean norm of translation along z
    r0 = np.sqrt(z**2)

    # Initialize a cache to store computed coefficients
    R_cache = {}

    # Precompute spherical Bessel functions
    jn_values = [spherical_jn(l_idx, k * r0) for l_idx in range(N_t)]

    # Function to compute translation coefficient recursively
    def compute_R(l_idx, n_idx):
        if l_idx == 0 and n_idx == 0:
            return 1.0 + 0j

        if (l_idx, n_idx) in R_cache:
            return R_cache[(l_idx, n_idx)]

        if l_idx == 0:
            value = jn_values[n_idx] * compute_R(l_idx, n_idx - 1) / (n_idx + 1)
        elif n_idx == 0:
            value = jn_values[l_idx] * compute_R(l_idx - 1, n_idx) / (l_idx + 1)
        else:
            term_a = jn_values[l_idx] * compute_R(l_idx - 1, n_idx) / (l_idx + 1)
            term_b = jn_values[n_idx] * compute_R(l_idx, n_idx - 1) / (n_idx + 1)
            value = (term_a + term_b) / 4  # Using a different division factor

        R_cache[(l_idx, n_idx)] = value
        return value

    # Compute the required translation coefficient
    return compute_R(l, n)


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




    def compute_euler_angles(Q):
        """Compute Euler angles from the rotation matrix using matrix logarithm."""
        A = logm(Q)  # Compute the matrix logarithm
        beta = np.arccos((np.trace(Q) - 1) / 2)
        if np.isclose(beta, 0):  # Handle gimbal lock
            alpha = np.arctan2(Q[0, 1], Q[0, 0])
            gamma = 0
        else:
            alpha = np.arctan2(A[1, 0], A[0, 0])
            gamma = np.arctan2(A[2, 1], A[2, 2])
        return alpha, beta, gamma

    # Calculate the Euler angles from the rotation matrix
    alpha, beta, gamma = compute_euler_angles(Q)

    # Calculate the Wigner D-matrix element using a different formula
    def wigner_d_element(l, mp, m, beta):
        # Use an alternative series expansion for the Wigner small d-matrix element
        d = 0
        for k in range(max(0, m-mp), min(l-mp, l+m)+1):
            coeff = ((-1)**k *
                     np.math.factorial(l+m) * np.math.factorial(l-m) *
                     np.math.factorial(l+mp) * np.math.factorial(l-mp))
            denom = (np.math.factorial(k) * np.math.factorial(l-mp-k) *
                     np.math.factorial(l+m-k) * np.math.factorial(k+mp-m))
            d += (coeff / denom) * (np.cos(beta / 2)**(2*l+m-mp-2*k)) * (np.sin(beta / 2)**(2*k+mp-m))
        return d

    # Calculate the Wigner D-matrix element
    d_element = wigner_d_element(n, v, m, beta)

    # Compute the phase factor using Euler angles
    phase_factor = np.exp(-1j * (m * alpha + v * gamma))

    # Calculate the spherical harmonics
    Y_m = sph_harm(m, n, 0, 0)
    Y_v = sph_harm(v, n, 0, 0)

    # Calculate the rotation coefficient T
    T = d_element * phase_factor * Y_v.conjugate() * Y_m

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
    # Calculate the magnitude and direction of r0
    r0_magnitude = np.linalg.norm(r0)
    theta = np.arccos(r0[2] / r0_magnitude)
    phi = np.arctan2(r0[1], r0[0])

    # Define wavevector k
    k = 2 * np.pi / wl

    # Use a Lie algebra-based method for rotation
    def lie_rotation_matrix(theta, phi):
        # Construct Lie algebra elements
        Jx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        Jy = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        # Exponentiate to get rotation matrices
        Rx = expm(Jx * theta)
        Ry = expm(Jy * phi)
        return Ry @ Rx

    # Generate Lie-based rotation matrix
    Q = lie_rotation_matrix(theta, phi)

    # Define a Chebyshev polynomial-based translation coefficient function
    def chebyshev_translation_coefficient(l, n, m, z, N_t):
        return np.polynomial.chebyshev.chebval(z / (l + 1), [0]*(l) + [1])

    # Rotate B using Lie-based rotation matrix
    rotated_B = np.zeros_like(B, dtype=complex)
    for l in range(N_t + 1):
        for v in range(-l, l + 1):
            for mp in range(-l, l + 1):
                rotated_B[l, v + N_t] += sph_harm(v, l, phi, theta).conjugate() * B[l, mp + N_t] * sph_harm(mp, l, phi, theta)

    # Apply Chebyshev polynomial-based translation along the new z direction
    translated_B = np.zeros_like(rotated_B, dtype=complex)
    for l in range(N_t + 1):
        for v in range(-l, l + 1):
            Rlnm_val = chebyshev_translation_coefficient(l, n, m, r0_magnitude, N_t)
            translated_B[l, v + N_t] = Rlnm_val * rotated_B[l, v + N_t]

    # Rotate back to the original orientation using the inverse of the Lie-based rotation matrix
    BR_nm = 0
    for l in range(N_t + 1):
        for mp in range(-l, l + 1):
            BR_nm += sph_harm(m, l, phi, theta) * translated_B[l, mp + N_t] * sph_harm(mp, l, phi, theta).conjugate()

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