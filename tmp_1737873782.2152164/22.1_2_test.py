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

    # Translation coefficient calculation using recursion
    # Initialize the array to store the coefficients
    Rlnm_coeffs = np.zeros((N_t, N_t), dtype=complex)

    # Base case for the recursion
    Rlnm_coeffs[0, 0] = np.exp(1j * k * z) * (1j)**(l-n)

    # Recurrence relations for the translation coefficients
    for l_prime in range(1, N_t):
        for n_prime in range(1, N_t):
            if l_prime >= n_prime:
                # Calculate the translation coefficient using recursion
                Rlnm_coeffs[l_prime, n_prime] = ((2 * l_prime - 1) / (l_prime - n_prime)) * (1j * k * z) * Rlnm_coeffs[l_prime - 1, n_prime - 1]
                if l_prime > n_prime:
                    Rlnm_coeffs[l_prime, n_prime] -= ((l_prime + n_prime - 1) / (l_prime - n_prime)) * Rlnm_coeffs[l_prime - 2, n_prime - 1]

    return Rlnm_coeffs[l, n]


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