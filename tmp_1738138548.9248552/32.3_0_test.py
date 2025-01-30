from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy
from scipy.constants import epsilon_0, c



def binding_force(P, phi, R, l, w, a, n):
    # Calculate polarizability using a different formula
    alpha = 4 * pi * epsilon_0 * a**3 * ((n**2 - 1) / (n**2 + 2))
    
    # Calculate the electric field amplitude from the power and beam waist
    E0 = np.sqrt(2 * P[0] / (pi * w**2 * epsilon_0))
    
    # Calculate the dipole moment induced in each sphere
    p = alpha * E0
    
    # Calculate the electric field at the location of the second sphere due to the first
    # Using a vector approach for the field calculation
    r_hat = np.array([np.cos(phi), np.sin(phi), 0])
    E_field = (1 / (4 * pi * epsilon_0)) * (3 * np.dot(p, r_hat) * r_hat - p) / R**3
    
    # Calculate the force on the second dipole
    F = np.dot(p, E_field)
    
    return F



def generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho):
    # Constants
    k = 2 * np.pi / l
    epsilon = 8.854187817e-12  # Permittivity of free space

    # Calculate polarizability using Clausius-Mossotti relation
    alpha = 4 * np.pi * epsilon * a**3 * ((n**2 - 1) / (n**2 + 2))

    # Initialize Hamiltonian matrix
    H = np.zeros((N, N), dtype=complex)

    # Compute interaction terms based on dipole-dipole interaction model
    for i in range(N):
        for j in range(N):
            if i == j:
                # Self-energy term, could include trap depth or other effects
                H[i, j] = P[i] / (epsilon * np.pi * w**2)
            else:
                # Interaction energy between different particles
                distance = np.abs(i - j) * R
                H[i, j] = (alpha**2 / (4 * np.pi * epsilon * distance**3)) * np.exp(-1j * k * distance)

    return H



# Background: The Runge-Kutta method is a powerful tool for numerically solving ordinary differential equations (ODEs). 
# In this context, we are dealing with the Lindblad master equation, which describes the time evolution of the density 
# matrix of an open quantum system. The correlation matrix C_ij = ⟨b_i^† b_j⟩ represents the expectation values of 
# phonon occupation numbers, where b_i and b_j are annihilation operators for phonons in the i-th and j-th modes, respectively.
# The Hamiltonian H describes the coherent evolution of the system, while the dissipation matrix L and the reservoir matrix M 
# account for the interaction with the environment. The fourth-order Runge-Kutta (RK4) method is used to integrate the 
# differential equations over time, providing a balance between accuracy and computational efficiency.




def runge_kutta(C0, H, L, M, t0, steps):
    '''Function to numerically solve the Lindblad master equation with the Runge-Kutta method.
    Input
    C0 : matrix of shape(N, N)
        Initial correlation matrix.
    H : matrix of shape(N, N)
        The Hamiltonian of the system.
    L : matrix of shape(N, N)
        The dissipation matrix.
    M : matrix of shape(N, N)
        The reservoir matrix.
    t0 : float
        The time point at which to calculate the phonon occupation.
    steps : int
        Number of simulation steps for the integration.
    Output
    nf : list of length N
        Phonon occupation of each trapped microsphere at time point `t0`.
    '''
    
    # Define the time step
    dt = t0 / steps
    
    # Function to compute the derivative of C
    def dCdt(C, H, L, M):
        # Commutator term: -i[H, C]
        commutator = -1j * (H @ C - C @ H)
        
        # Dissipative terms: LCL^† - 0.5{L^†L, C}
        dissipative = L @ C @ L.conj().T - 0.5 * (L.conj().T @ L @ C + C @ L.conj().T @ L)
        
        # Reservoir interaction term: M
        reservoir = M
        
        # Total derivative
        return commutator + dissipative + reservoir
    
    # Initialize the correlation matrix
    C = C0
    
    # Runge-Kutta 4th order integration
    for _ in range(steps):
        k1 = dCdt(C, H, L, M)
        k2 = dCdt(C + 0.5 * dt * k1, H, L, M)
        k3 = dCdt(C + 0.5 * dt * k2, H, L, M)
        k4 = dCdt(C + dt * k3, H, L, M)
        
        # Update C using the RK4 formula
        C += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Calculate the phonon occupation numbers (diagonal elements of C)
    nf = np.diag(C).real
    
    return nf


try:
    targets = process_hdf5_to_tuple('32.3', 4)
    target = targets[0]
    n0 = [39549953.17, 197.25, 197.25, 197.25, 197.25]
    Gamma = 0.001
    P = [100e-3, 100e-3, 100e-3, 100e-3, 100e-3]
    phi = np.pi / 2
    R = 0.99593306197 * 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    h = 1e-6
    N = np.size(P)
    rho = 2.648e3
    C0 = np.diag(n0)
    H = generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho)
    L = - Gamma * np.identity(N) / 2
    M = 197.25 * Gamma * np.identity(N) / 2
    t0 = 0.02
    steps = 100000
    assert np.allclose(runge_kutta(C0, H, L, M, t0, steps), target)

    target = targets[1]
    n0 = [197.25, 39549953.17, 39549953.17, 39549953.17, 39549953.17]
    Gamma = 0.001
    P = [100e-3, 100e-3, 100e-3, 100e-3, 100e-3]
    phi = np.pi / 2
    R = 0.99593306197 * 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    h = 1e-6
    N = np.size(P)
    rho = 2.648e3
    C0 = np.diag(n0)
    H = generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho)
    L = - Gamma * np.identity(N) / 2
    M = 197.25 * Gamma * np.identity(N) / 2
    t0 = 0.05
    steps = 100000
    assert np.allclose(runge_kutta(C0, H, L, M, t0, steps), target)

    target = targets[2]
    n0 = [39549953.17, 197.25, 197.25, 197.25, 197.25]
    Gamma = 0.001
    P = [100e-3, 100e-3, 100e-3, 100e-3, 100e-3]
    phi = np.pi / 2
    R = 0.99593306197 * 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    h = 1e-6
    N = np.size(P)
    rho = 2.648e3
    C0 = np.diag(n0)
    H = generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho)
    L = - Gamma * np.identity(N) / 2
    M = 197.25 * Gamma * np.identity(N) / 2
    t0 = 0.05
    steps = 100000
    assert np.allclose(runge_kutta(C0, H, L, M, t0, steps), target)

    target = targets[3]
    n0 = [197.25, 197.25, 39549953.17, 197.25, 197.25]
    Gamma = 0
    P = [100e-3, 100e-3, 100e-3, 100e-3, 100e-3]
    phi = np.pi / 2
    R = 0.99593306197 * 1550e-9
    l = 1550e-9
    w = 600e-9
    a = 100e-9
    n = 1.444
    h = 1e-6
    N = np.size(P)
    rho = 2.648e3
    C0 = np.diag(n0)
    H = generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho)
    L = - Gamma * np.identity(N) / 2
    M = 197.25 * Gamma * np.identity(N) / 2
    t0 = 0.02
    steps = 100000
    nf = runge_kutta(C0, H, L, M, t0, steps)
    diff = sum(nf) - sum(n0)
    def is_symmetric(array, rtol=1e-05, atol=1e-08):
        return np.all(np.isclose(array, array[::-1], rtol=rtol, atol=atol))
    assert (abs(diff)<1e-6, is_symmetric(nf)) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e