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




    def matrix_logarithm(Q):
        """Compute the matrix logarithm using its series expansion."""
        I = np.eye(3)
        A = Q - I
        term = A
        result = term
        for i in range(2, 30):  # Convergence for matrix logarithm
            term = (-1)**(i+1) * np.dot(A, term) / i
            result += term
        return result

    def compute_axis_angle(Q):
        """Extract axis-angle representation from the rotation matrix."""
        A = matrix_logarithm(Q)
        angle = np.linalg.norm([A[2, 1], A[0, 2], A[1, 0]])
        axis = np.array([A[2, 1], A[0, 2], A[1, 0]]) / angle if angle != 0 else np.array([0, 0, 1])
        return axis, angle

    # Compute the axis and angle of rotation
    axis, angle = compute_axis_angle(Q)

    # Calculate Euler angles from axis-angle representation
    alpha = np.arctan2(axis[1], axis[0]) if axis[0] != 0 or axis[1] != 0 else 0
    beta = angle
    gamma = np.arctan2(axis[2], np.sqrt(axis[0]**2 + axis[1]**2))

    # Calculate the Wigner D-matrix element using a novel approach
    def wigner_d_element(n, v, m, beta):
        # Use a novel approach to calculate the Wigner small d-matrix element
        d = np.cos(beta / 2)**(m - v) * np.sin(beta / 2)**(v + m)
        return d

    # Calculate the Wigner D-matrix element
    d_element = wigner_d_element(n, v, m, beta)

    # Compute the exponential phase factor
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