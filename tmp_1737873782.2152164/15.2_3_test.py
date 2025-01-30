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
    A, B: A and B matrices; 2D arrays of dimension N-1 by N-1 where each element is a float
    '''

    # Constants
    m = 9.10938356e-31  # mass of electron in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in J.s
    dx = L / N  # space step

    # Coefficients
    alpha = hbar**2 / (2 * m * dx**2)
    beta = 1j * hbar / (2 * h)

    # Initialize A and B matrices
    A = np.zeros((N-1, N-1), dtype=complex)
    B = np.zeros((N-1, N-1), dtype=complex)

    # Fill the matrices
    for i in range(N-1):
        A[i, i] = 1 + beta + alpha * h
        B[i, i] = 1 - beta - alpha * h
        if i > 0:
            A[i, i-1] = -alpha * h / 2
            B[i, i-1] = alpha * h / 2
        if i < N-2:
            A[i, i+1] = -alpha * h / 2
            B[i, i+1] = alpha * h / 2

    return A, B





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
    m = 9.10938356e-31  # mass of electron in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in J.s
    dx = L / N  # space step
    h = T / nstep  # time step

    # Initialize matrices A and B
    A, B = init_AB(N, L, h)

    # Initialize Gaussian wave packet
    x = np.linspace(0, L, N+1)
    x0 = L / 2  # Center of the well
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)
    
    # Enforce boundary conditions: psi[0] = psi[N] = 0
    psi[0] = psi[N] = 0

    # Time evolution
    for _ in range(nstep):
        # Right-hand side B * psi
        b = B @ psi[1:N]
        
        # Solve A * psi_new = b
        psi_new = linalg.solve(A, b)
        
        # Update psi with boundary conditions
        psi[1:N] = psi_new

    # Return the real part of the wavefunction
    return np.real(psi)


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e