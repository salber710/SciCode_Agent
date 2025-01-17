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



# Background: 
# The Crank-Nicolson method is used to solve the time-dependent Schrödinger equation by discretizing time and space.
# The method involves solving a system of linear equations at each time step, which can be expressed as A * psi(t+h) = B * psi(t).
# The initial wavefunction is a Gaussian wave packet, which is a common choice for simulating quantum particles.
# The wave packet is defined by its width (sigma) and wave number (kappa), and it is centered in the potential well.
# The boundary conditions require the wavefunction to be zero at the edges of the well.
# The matrices A and B are tridiagonal and are used to evolve the wavefunction in time.
# The solution involves iteratively solving the linear system for each time step to update the wavefunction.



def crank_nicolson(sigma, kappa, T, nstep, N, L):
    '''Solve the Crank-Nicolson equation of the form A * psi(x, t+h) = B * psi(x, t)
    Input
    sigma: the sigma parameter of a Gaussian wave packet; float
    kappa: the kappa parameter of a Gaussian wave packet; float
    T: the total amount of time for the evolution in seconds; float
    nstep: the total number of time steps; int
    N: the total number of grid intervals; int
    L: the dimension of the 1D well in meters; float
    Output
    psi: the real part of the wavefunction after time T; 1D array of float with shape (N+1,)
    '''
    
    # Constants
    m = 9.109e-31  # electron mass in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in Js

    # Spatial discretization
    dx = L / N  # grid spacing
    x = np.linspace(0, L, N+1)  # grid points

    # Time step size
    h = T / nstep

    # Initialize the Gaussian wave packet
    x0 = L / 2  # center of the well
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)
    psi[0] = psi[-1] = 0  # apply boundary conditions

    # Initialize matrices A and B
    alpha = 1j * hbar * h / (4 * m * dx**2)
    beta = 1 + 2 * alpha
    gamma = 1 - 2 * alpha

    diagonals_A = np.zeros((3, N-1), dtype=complex)
    diagonals_B = np.zeros((3, N-1), dtype=complex)

    diagonals_A[0, :] = -alpha  # upper diagonal
    diagonals_A[1, :] = beta    # main diagonal
    diagonals_A[2, :] = -alpha  # lower diagonal

    diagonals_B[0, :] = alpha   # upper diagonal
    diagonals_B[1, :] = gamma   # main diagonal
    diagonals_B[2, :] = alpha   # lower diagonal

    A = sparse.diags(diagonals_A, offsets=[-1, 0, 1], format='csr')
    B = sparse.diags(diagonals_B, offsets=[-1, 0, 1], format='csr')

    # Time evolution
    for _ in range(nstep):
        # Solve the linear system A * psi_new = B * psi_old
        psi_interior = psi[1:-1]  # exclude boundary points
        rhs = B.dot(psi_interior)
        psi_new_interior = linalg.spsolve(A, rhs)
        psi[1:-1] = psi_new_interior

    # Return the real part of the wavefunction
    psi_real = np.real(psi)
    return psi_real


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('15.2', 4)
target = targets[0]

sigma = 1e-10
kappa = 5e10
T=9e-16
h=5e-18
nstep=int(T/h)
N=200
L=1e-8
assert np.allclose(crank_nicolson(sigma, kappa, T, nstep, N, L), target)
target = targets[1]

sigma = 1e-10
kappa = 1e10
T=1e-14
h=5e-18
nstep=int(T/h)
N=200
L=2e-8
assert np.allclose(crank_nicolson(sigma, kappa, T, nstep, N, L), target)
target = targets[2]

sigma = 2e-10
kappa = 5e10
T=1e-14
h=5e-18
nstep=int(T/h)
N=300
L=1e-7
assert np.allclose(crank_nicolson(sigma, kappa, T, nstep, N, L), target)
target = targets[3]

sigma = 2e-10
kappa = 0
T=1e-14
h=5e-18
nstep=int(T/h)
N=200
L=2e-8
wave = crank_nicolson(sigma, kappa, T, nstep, N, L)
assert np.allclose(wave[:wave.shape[0]//2][::-1],wave[wave.shape[0]//2+1:]) == target
