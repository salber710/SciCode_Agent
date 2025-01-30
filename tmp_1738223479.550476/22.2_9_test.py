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


try:
    targets = process_hdf5_to_tuple('22.2', 3)
    target = targets[0]
    Q = np.matrix([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])
    assert np.allclose(Tnvm(2, 1, 1, Q), target)

    target = targets[1]
    Q = np.matrix([[0, 0, 1],[0, 1, 0],[-1, 0, 0]])
    assert np.allclose(Tnvm(5, 2, 1, Q), target)

    target = targets[2]
    Q = np.matrix([[- 1 / np.sqrt(2), - 1 / np.sqrt(2), 0],[1 / (2 * np.sqrt(2)), - 1 / (2 * np.sqrt(2)), np.sqrt(3) / 2],[- np.sqrt(3 / 2) / 2, np.sqrt(3 / 2) / 2, 1 / 2]])
    assert np.allclose(Tnvm(1, 1, 0, Q), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e