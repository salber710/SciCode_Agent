import numpy as np
from scipy import linalg, sparse



# Background: 
# The Crank-Nicolson method is a numerical technique used to solve differential equations such as the time-dependent Schrödinger equation. 
# This method is implicit and unconditionally stable, making it suitable for solving parabolic PDEs. 
# For the 1D time-dependent Schrödinger equation, the method involves discretizing both time and space.
# In the context of quantum mechanics, the wave function ψ(x, t) evolves over time, and we can represent this evolution as a matrix equation.
# The matrices A and B are derived from the Crank-Nicolson discretization:
#   A = (I + i(h/2)H/ħ), B = (I - i(h/2)H/ħ)
# Where I is the identity matrix, H is the Hamiltonian matrix, i is the imaginary unit, h is the time step, and ħ is the reduced Planck's constant.
# For a particle in a 1D potential well, the Hamiltonian H in a discretized form is a tridiagonal matrix given by:
#   H[i, i] = 2, H[i, i+1] = -1, H[i, i-1] = -1 (for i in 1 to N-2, considering 0-based index)
# The spatial grid is divided into N intervals, and the wave function is evaluated at N+1 points (including boundary points).
# Boundary conditions are ψ(0, t) = ψ(L, t) = 0, hence we consider only the inner N-1 grid points for evolution.



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

    # Spatial step size
    dx = L / N

    # Prefactor for the Hamiltonian matrix
    prefactor = -hbar**2 / (2 * m * dx**2)

    # Time step factor
    factor = 1j * h / (2 * hbar)

    # Hamiltonian matrix (tridiagonal)
    main_diag = np.full(N-1, 2)  # Main diagonal
    off_diag = np.full(N-2, -1)  # Off diagonals

    # Create sparse tridiagonal Hamiltonian matrix
    H = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N-1, N-1)).toarray()

    # Scale the Hamiltonian with the prefactor
    H *= prefactor

    # Identity matrix
    I = np.eye(N-1)

    # A and B matrices
    A = I + factor * H
    B = I - factor * H

    return A, B

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('15.1', 3)
target = targets[0]

assert np.allclose(init_AB(2, 1e-7, 1e-18), target)
target = targets[1]

assert np.allclose(init_AB(4, 1e-7, 1e-18), target)
target = targets[2]

assert (init_AB(5, 1e-8, 1e-18)[0].shape==(4,4)) == target
