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



    # Decompose the rotation matrix Q into its quaternion representation
    def rotation_matrix_to_quaternion(Q):
        # Assuming Q is a valid rotation matrix
        trace = np.trace(Q)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (Q[2, 1] - Q[1, 2]) / S
            qy = (Q[0, 2] - Q[2, 0]) / S
            qz = (Q[1, 0] - Q[0, 1]) / S
        elif (Q[0, 0] > Q[1, 1]) and (Q[0, 0] > Q[2, 2]):
            S = np.sqrt(1.0 + Q[0, 0] - Q[1, 1] - Q[2, 2]) * 2
            qw = (Q[2, 1] - Q[1, 2]) / S
            qx = 0.25 * S
            qy = (Q[0, 1] + Q[1, 0]) / S
            qz = (Q[0, 2] + Q[2, 0]) / S
        elif Q[1, 1] > Q[2, 2]:
            S = np.sqrt(1.0 + Q[1, 1] - Q[0, 0] - Q[2, 2]) * 2
            qw = (Q[0, 2] - Q[2, 0]) / S
            qx = (Q[0, 1] + Q[1, 0]) / S
            qy = 0.25 * S
            qz = (Q[1, 2] + Q[2, 1]) / S
        else:
            S = np.sqrt(1.0 + Q[2, 2] - Q[0, 0] - Q[1, 1]) * 2
            qw = (Q[1, 0] - Q[0, 1]) / S
            qx = (Q[0, 2] + Q[2, 0]) / S
            qy = (Q[1, 2] + Q[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    # Convert rotation matrix to quaternion
    q = rotation_matrix_to_quaternion(Q)

    # Calculate Euler angles from quaternion
    def quaternion_to_euler(q):
        qw, qx, qy, qz = q
        t0 = +2.0 * (qw * qx + qy * qz)
        t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
        alpha = np.arctan2(t0, t1)

        t2 = +2.0 * (qw * qy - qz * qx)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        beta = np.arcsin(t2)

        t3 = +2.0 * (qw * qz + qx * qy)
        t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
        gamma = np.arctan2(t3, t4)
        
        return alpha, beta, gamma

    alpha, beta, gamma = quaternion_to_euler(q)

    # Calculate the Wigner D-matrix element using quaternion-derived angles
    def wigner_d_quaternion(l, mp, m, beta):
        # Use quaternion-derived angles to compute the Wigner d-matrix element
        return np.cos(beta / 2)**(m - mp) * np.sin(beta / 2)**(mp + m)

    # Calculate the Wigner D-matrix element
    d_element = wigner_d_quaternion(n, v, m, beta)

    # Compute the phase factor
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