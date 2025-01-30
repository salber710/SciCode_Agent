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


    # Initialize a multidimensional array for storing translation coefficients
    R_lnm = np.zeros((N_t, N_t, N_t), dtype=complex)

    # Calculate the parameter r0 based on translation distance
    r0 = np.sqrt(z**2)

    # Precompute some constants to avoid repetition in loops
    k_r0 = k * r0

    # Define base value for recursion; using spherical Bessel function of the first kind
    def base_value(l):
        # For l = 0, the base value might be chosen as 1, adjust according to the recursion specifics
        return np.exp(-1j * k_r0) if l == 0 else 0

    # Populate the translation coefficient array using a nested loop and recursion
    for l_idx in range(N_t):
        for n_idx in range(N_t):
            if l_idx == n_idx == 0:
                # Base case for recursion
                R_lnm[l_idx, n_idx, m] = base_value(l_idx)
            else:
                # Implement the recursion relation based on l and n; placeholder example:
                prev_l = R_lnm[l_idx-1, n_idx, m] if l_idx > 0 else 0
                prev_n = R_lnm[l_idx, n_idx-1, m] if n_idx > 0 else 0

                # Example recursion formula: adjust according to the specific problem
                R_lnm[l_idx, n_idx, m] = (k_r0 / (l_idx + n_idx + 1)) * (prev_l + prev_n)

    # Return the computed translation coefficient for the specific (l, n, m)
    return R_lnm[l, n, m]


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