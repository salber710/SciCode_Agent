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



    def compute_wigner_d(j, mp, m, beta):
        # Calculate Wigner d-matrix element using a direct summation formula
        d = 0
        for k in range(max(0, m-mp), min(j-mp, j+m)+1):
            coeff = ((-1)**(k+mp-m) * 
                     np.sqrt(factorial(j+mp) * factorial(j-mp) * factorial(j+m) * factorial(j-m)) /
                     (factorial(k) * factorial(j-mp-k) * factorial(j+m-k) * factorial(k+mp-m)))
            d += coeff * (np.cos(beta/2)**(2*j+mp-m-2*k)) * (np.sin(beta/2)**(2*k+mp-m))
        return d

    def extract_euler_angles(Q):
        # Compute Euler angles from the rotation matrix Q
        beta = np.arccos(Q[2, 2])
        alpha = np.arctan2(Q[2, 1], Q[2, 0])
        gamma = np.arctan2(Q[1, 2], -Q[0, 2])
        return alpha, beta, gamma

    # Extract Euler angles from the rotation matrix Q
    alpha, beta, gamma = extract_euler_angles(Q)

    # Compute the Wigner d-matrix element for the given n, v, m and beta
    d_element = compute_wigner_d(n, v, m, beta)

    # Compute the exponential phase factor
    phase_factor = np.exp(-1j * (m * alpha + v * gamma))

    # Calculate the rotation coefficient T
    T = d_element * phase_factor

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