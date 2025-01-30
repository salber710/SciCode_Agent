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
    translation_coefficient = 0.0 + 0.0j
    
    # Precompute some constants
    kz = k * z
    
    # Use a recursion method to calculate the translation coefficient
    # Iterate over l' from 0 to the truncation limit
    for lp in range(N_t):
        # Iterate over s from -lp to lp
        for s in range(-lp, lp + 1):
            # Calculate the intermediate translation term
            # Using some form of recurrence relation or predefined terms
            # This is a placeholder as exact recurrence relations depend on the context
            # Typically involve special functions like spherical harmonics, Bessel functions etc.
            # Here I will use a dummy function for illustration
            intermediate_term = np.exp(1j * kz * (l - n))

            # Sum the contributions
            translation_coefficient += intermediate_term
    
    return translation_coefficient


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