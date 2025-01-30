from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import linalg, sparse





def init_AB(N, L, h):
    '''Initialize the matrices A and B
    Input
    N: the number of grid intervals; int
    L: the dimension of the 1D well; float
    h: the size of each time step in seconds; float
    Output
    A,B: A and B matrices; 2D arrays of dimension N-1 by N-1 where each element is a float
    '''

    # Constants
    m = 9.109e-31  # electron mass in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in Js
    dx = L / N  # spatial grid size

    # Coefficients for the matrices
    alpha = (1j * hbar * h) / (4 * m * dx**2)
    beta = 1j * hbar * h / 2

    # Initialize the diagonals
    main_diag = np.full(N-1, 1 + 2 * alpha)
    off_diag = np.full(N-2, -alpha)

    # Construct matrices A and B
    A = sparse.diags([main_diag, off_diag, off_diag], [0, -1, 1], format='csr')
    B = sparse.diags([main_diag.conjugate(), -off_diag.conjugate(), -off_diag.conjugate()], [0, -1, 1], format='csr')

    return A.toarray(), B.toarray()


try:
    targets = process_hdf5_to_tuple('15.1', 3)
    target = targets[0]
    assert np.allclose(init_AB(2, 1e-7, 1e-18), target)

    target = targets[1]
    assert np.allclose(init_AB(4, 1e-7, 1e-18), target)

    target = targets[2]
    assert (init_AB(5, 1e-8, 1e-18)[0].shape==(4,4)) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e