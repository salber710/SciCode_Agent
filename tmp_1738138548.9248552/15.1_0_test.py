from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import linalg, sparse



# Background: 
# The Crank-Nicolson method is a numerical technique used to solve partial differential equations like the time-dependent Schrödinger equation. 
# It is an implicit method that is unconditionally stable and second-order accurate in both time and space. 
# When applied to the Schrödinger equation, it involves discretizing the spatial domain into a grid and using finite differences to approximate derivatives.
# The method results in a system of linear equations that can be expressed in matrix form: A * psi(t+h) = B * psi(t).
# Here, A and B are tridiagonal matrices that arise from the discretization process.
# The entries of these matrices are determined by the coefficients of the finite difference approximations.
# For a 1D potential well, the wavefunction psi must be zero at the boundaries, which means we only evolve the interior points.
# The matrices A and B will have dimensions (N-1) x (N-1) where N is the number of grid intervals.
# The diagonal and off-diagonal elements of these matrices are derived from the discretized Schrödinger equation.
# The constants involved include the electron mass m and the reduced Planck's constant ħ.



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

    # Spatial discretization
    dx = L / N  # grid spacing

    # Coefficients for the tridiagonal matrices
    alpha = 1j * hbar * h / (4 * m * dx**2)
    beta = 1j * hbar / (2 * h)

    # Initialize the diagonals
    main_diag_A = (1 + 2 * alpha) * np.ones(N-1)
    off_diag_A = -alpha * np.ones(N-2)

    main_diag_B = (1 - 2 * alpha) * np.ones(N-1)
    off_diag_B = alpha * np.ones(N-2)

    # Create sparse tridiagonal matrices A and B
    A = sparse.diags([off_diag_A, main_diag_A, off_diag_A], [-1, 0, 1], format='csr')
    B = sparse.diags([off_diag_B, main_diag_B, off_diag_B], [-1, 0, 1], format='csr')

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