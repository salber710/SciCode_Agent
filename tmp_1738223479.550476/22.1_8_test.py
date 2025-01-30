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



    # Calculate r0 using the absolute value of z
    r0 = np.abs(z)

    # Initialize a dictionary to store translation coefficients
    R_lnm = {}

    # Precompute spherical Bessel functions for efficiency
    jn_values = [spherical_jn(l_idx, k * r0) for l_idx in range(N_t)]

    def recursive_R(l_idx, n_idx):
        # Base case
        if l_idx == 0 and n_idx == 0:
            return 1 + 0j

        # Check if the value is already computed
        if (l_idx, n_idx) in R_lnm:
            return R_lnm[(l_idx, n_idx)]

        if l_idx == 0:
            value = jn_values[n_idx] * recursive_R(l_idx, n_idx - 1) / (1 + n_idx**2)
        elif n_idx == 0:
            value = jn_values[l_idx] * recursive_R(l_idx - 1, n_idx) / (1 + l_idx**2)
        else:
            # Use a different recursive relation with adjusted weights
            term_a = jn_values[l_idx] * recursive_R(l_idx - 1, n_idx) / ((l_idx + 1) * (n_idx + 1))
            term_b = jn_values[n_idx] * recursive_R(l_idx, n_idx - 1) / ((l_idx + 1) * (n_idx + 1))
            value = (term_a + term_b) / 3

        # Store the computed value
        R_lnm[(l_idx, n_idx)] = value
        return value

    # Compute the required translation coefficient
    return recursive_R(l, n)


try:
    targets = process_hdf5_to_tuple('22.1', 3)
    target = targets[0]
    l = 1
    n = 1
    m = 0
    wl = 2 * np.pi
    k = 2 * np.pi / wl
    z = 0.75
    N_t = 8
    assert np.allclose(Rlnm(l, n, m, k, z, N_t), target)

    target = targets[1]
    l = 2
    n = 1
    m = 0
    wl = 2 * np.pi
    k = 2 * np.pi / wl
    z = 0.75
    N_t = 8
    assert np.allclose(Rlnm(l, n, m, k, z, N_t), target)

    target = targets[2]
    l = 3
    n = 2
    m = 1
    wl = 2 * np.pi
    k = 2 * np.pi / wl
    z = 0.75
    N_t = 8
    assert np.allclose(Rlnm(l, n, m, k, z, N_t), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e