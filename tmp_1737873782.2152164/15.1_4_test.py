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
    dx = L / N  # spatial step size

    # Coefficient for the off-diagonal elements
    coeff = hbar**2 / (2 * m * dx**2)

    # Coefficient for the time step
    r = 1j * h * coeff / hbar

    # Initialize A and B as zero matrices of size (N-1)x(N-1)
    A = np.zeros((N-1, N-1), dtype=complex)
    B = np.zeros((N-1, N-1), dtype=complex)

    # Fill the matrices A and B
    for i in range(N-1):
        A[i, i] = 1 + r
        B[i, i] = 1 - r
        if i > 0:
            A[i, i-1] = -r / 2
            B[i, i-1] = r / 2
        if i < N-2:
            A[i, i+1] = -r / 2
            B[i, i+1] = r / 2

    return A, B


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