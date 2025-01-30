from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy import linalg, sparse


def init_AB(N, L, h):
    m = 9.109e-31  # electron mass in kg
    hbar = 1.0545718e-34  # reduced Planck's constant in Js
    dx = L / N  # grid spacing

    alpha = 1j * hbar * h / (4 * m * dx**2)

    # Initialize matrices A and B using a single loop for a compact form
    A = np.zeros((N-1, N-1), dtype=complex)
    B = np.zeros((N-1, N-1), dtype=complex)

    for i in range(N-1):
        A[i, i] = 1 + 2 * alpha
        B[i, i] = 1 - 2 * alpha
        if i < N-2:
            A[i, i+1] = A[i+1, i] = -alpha
            B[i, i+1] = B[i+1, i] = alpha

    return A, B



# Background: The Crank-Nicolson method is a numerical technique used to solve partial differential equations, such as the time-dependent Schrödinger equation. It is an implicit method that is unconditionally stable and second-order accurate in both time and space. The method involves solving a system of linear equations at each time step, which can be expressed in the form A * psi(x, t+h) = B * psi(x, t), where A and B are matrices derived from the discretization of the Schrödinger equation. The initial wavefunction is a Gaussian wave packet, which is a common choice for simulating quantum particles due to its localized nature and well-defined momentum. The wave packet is evolved over time using the Crank-Nicolson method, and the real part of the wavefunction is extracted after the evolution.

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

    # Grid setup
    dx = L / N
    x = np.linspace(0, L, N+1)
    x0 = L / 2  # Center of the Gaussian wave packet

    # Time step
    h = T / nstep

    # Initialize the Gaussian wave packet
    psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)

    # Apply boundary conditions
    psi[0] = psi[-1] = 0

    # Initialize matrices A and B
    alpha = 1j * hbar * h / (4 * m * dx**2)
    A = np.zeros((N-1, N-1), dtype=complex)
    B = np.zeros((N-1, N-1), dtype=complex)

    for i in range(N-1):
        A[i, i] = 1 + 2 * alpha
        B[i, i] = 1 - 2 * alpha
        if i < N-2:
            A[i, i+1] = A[i+1, i] = -alpha
            B[i, i+1] = B[i+1, i] = alpha

    # Time evolution
    for _ in range(nstep):
        # Extract the inner part of psi (excluding boundary points)
        psi_inner = psi[1:N]

        # Solve the linear system A * psi_new = B * psi_inner
        rhs = B @ psi_inner
        psi_new_inner = linalg.solve(A, rhs)

        # Update psi with the new values
        psi[1:N] = psi_new_inner

    # Return the real part of the wavefunction
    psi_real = np.real(psi)
    return psi_real


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