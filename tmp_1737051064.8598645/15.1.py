import numpy as np
from scipy import linalg, sparse



# Background: 
# The Crank-Nicolson method is a numerical technique used to solve partial differential equations, such as the time-dependent Schrödinger equation. 
# It is an implicit method that is unconditionally stable and second-order accurate in both time and space. 
# When applied to the Schrödinger equation, it involves discretizing the spatial domain into a grid and using finite differences to approximate derivatives.
# The method results in a system of linear equations that can be expressed in the form A * psi(t+h) = B * psi(t), where A and B are tridiagonal matrices.
# For a 1D potential well, the wavefunction psi(x) must be zero at the boundaries, which means we only evolve the interior points.
# The matrices A and B are constructed based on the discretization of the spatial domain and the time-stepping parameter h.
# The entries of these matrices are determined by the finite difference approximation of the second derivative and the time-stepping scheme.
# The electron mass m and the reduced Planck's constant ħ are used to calculate the kinetic energy term in the Hamiltonian.



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
    beta = 1 + 2 * alpha
    gamma = 1 - 2 * alpha

    # Initialize A and B as sparse matrices
    diagonals_A = np.zeros((3, N-1), dtype=complex)
    diagonals_B = np.zeros((3, N-1), dtype=complex)

    # Fill the diagonals
    diagonals_A[0, :] = -alpha  # upper diagonal
    diagonals_A[1, :] = beta    # main diagonal
    diagonals_A[2, :] = -alpha  # lower diagonal

    diagonals_B[0, :] = alpha   # upper diagonal
    diagonals_B[1, :] = gamma   # main diagonal
    diagonals_B[2, :] = alpha   # lower diagonal

    # Create sparse matrices A and B
    A = sparse.diags(diagonals_A, offsets=[-1, 0, 1], format='csr')
    B = sparse.diags(diagonals_B, offsets=[-1, 0, 1], format='csr')

    return A, B


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('15.1', 3)
target = targets[0]

assert np.allclose(init_AB(2, 1e-7, 1e-18), target)
target = targets[1]

assert np.allclose(init_AB(4, 1e-7, 1e-18), target)
target = targets[2]

assert (init_AB(5, 1e-8, 1e-18)[0].shape==(4,4)) == target
