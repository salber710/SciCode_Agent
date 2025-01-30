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



    # Initialize a dictionary to store the translation coefficients
    R_coefficients = {}

    # Calculate the parameter r0 as the absolute value of z
    r0 = abs(z)

    # Precompute the spherical Bessel functions for speed
    jn_values = [scipy.special.spherical_jn(l_idx, k * r0) for l_idx in range(N_t)]

    # Base case function
    def base_case(l_idx, n_idx):
        if l_idx == 0 and n_idx == 0:
            return 1.0
        return 0.0

    # Recursive function using memoization
    def compute_R(l_idx, n_idx):
        if (l_idx, n_idx, m) in R_coefficients:
            return R_coefficients[(l_idx, n_idx, m)]
        
        if l_idx == 0 or n_idx == 0:
            # If either index is zero, use the base case
            result = base_case(l_idx, n_idx)
        else:
            # Recursive relation, using spherical Bessel function values
            term1 = (2 * l_idx - 1) / l_idx * jn_values[l_idx] * compute_R(l_idx - 1, n_idx)
            term2 = (2 * n_idx - 1) / n_idx * jn_values[n_idx] * compute_R(l_idx, n_idx - 1)
            result = 0.5 * (term1 + term2)

        R_coefficients[(l_idx, n_idx, m)] = result
        return result

    # Compute the required translation coefficient
    return compute_R(l, n)


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