import numpy as np
from scipy import linalg, sparse

# Background: The Crank-Nicolson method is a numerical technique used to solve partial differential equations, such as the time-dependent Schrödinger equation. 
# It is an implicit method that is unconditionally stable and second-order accurate in both time and space. 
# When applied to the Schrödinger equation, it involves discretizing the spatial domain into a grid and using finite differences to approximate derivatives.
# The method results in a system of linear equations that can be expressed in matrix form: A * psi(x, t+h) = B * psi(x, t).
# Here, A and B are tridiagonal matrices that arise from the discretization process. 
# The entries of these matrices are determined by the coefficients of the finite difference approximations.
# For a 1D potential well, the wavefunction must be zero at the boundaries, so the matrices A and B are of size (N-1) x (N-1), where N is the number of grid intervals.
# The mass of the electron (m) and the reduced Planck's constant (hbar) are used to calculate the coefficients in the matrices.



def init_AB(N, L, h):
    '''Initialize the matrices A and B
    Input
    N: the number of grid intervals; int
    L: the dimension of the 1D well; float
    h: the size of each time step in seconds; float
    Output
    A,B: A and B matrices; 2D arrays of dimension N-1 by N-1 where each element is a float
    '''
    if N <= 0:
        raise ValueError("Number of grid intervals N must be positive.")
    if L < 0:
        raise ValueError("Dimension of the 1D well L must be non-negative.")

    # Constants
    m = 9.109e-31  # mass of electron in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in Js

    # Spatial step size
    dx = L / N if N > 0 else 0  # Avoid division by zero

    # Coefficients for the tridiagonal matrices
    r = hbar * h / (4 * m * dx**2) if dx != 0 else 0  # Avoid division by zero

    # Diagonal and off-diagonal elements for A and B
    diag_A = (1 + 2 * r) * np.ones(N-1)
    off_diag_A = -r * np.ones(N-2)

    diag_B = (1 - 2 * r) * np.ones(N-1)
    off_diag_B = r * np.ones(N-2)

    # Construct sparse tridiagonal matrices A and B
    A = sparse.diags([off_diag_A, diag_A, off_diag_A], [-1, 0, 1]).toarray()
    B = sparse.diags([off_diag_B, diag_B, off_diag_B], [-1, 0, 1]).toarray()

    # Ensure matrices are symmetric if h is zero
    if h == 0:
        A = np.eye(N-1)
        B = np.eye(N-1)

    return A, B



# Background: The Crank-Nicolson method is used to solve the time-dependent Schrödinger equation, which describes the evolution of quantum states over time. 
# In this context, we are dealing with a Gaussian wave packet, which is a common initial condition in quantum mechanics due to its well-defined position and momentum.
# The wave packet is defined by its width (sigma) and wave number (kappa), and it evolves over time according to the Schrödinger equation.
# The Crank-Nicolson method involves solving a system of linear equations at each time step, which is represented by the matrices A and B.
# The initial wavefunction is set up according to the Gaussian form, and the boundary conditions ensure that the wavefunction is zero at the edges of the potential well.
# The task is to compute the wavefunction at each grid point after a specified total time T, using nstep time steps.



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
    m = 9.109e-31  # mass of electron in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in Js

    # Time step size
    h = T / nstep

    # Spatial step size
    dx = L / N

    # Initialize the matrices A and B
    A, B = init_AB(N, L, h)

    # Initialize the Gaussian wave packet
    x = np.linspace(0, L, N+1)
    x0 = L / 2  # Center of the well
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)

    # Apply boundary conditions: psi[0] = psi[N] = 0
    psi[0] = 0
    psi[-1] = 0

    # Time evolution
    for _ in range(nstep):
        # Solve the linear system A * psi_new = B * psi_old
        psi_inner = psi[1:-1]  # Exclude boundary points
        b = B @ psi_inner
        psi_new_inner = linalg.solve(A, b)
        
        # Update psi with the new values
        psi[1:-1] = psi_new_inner

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
