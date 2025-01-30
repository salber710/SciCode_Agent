from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy
from scipy.constants import epsilon_0, c



def binding_force(P, phi, R, l, w, a, n):
    '''Function to calculate the optical binding force between two trapped nanospheres.
    Input
    P : list of length 2
        Power of the two optical traps.
    phi : float
        Polarization direction of the optical traps.
    R : float
        Distance between the trapped nanospheres.
    l : float
        Wavelength of the optical traps.
    w : float
        Beam waist of the optical traps.
    a : float
        Radius of the trapped microspheres.
    n : float
        Refractive index of the trapped microspheres.
    Output
    F : float
        The optical binding force between two trapped nanospheres.
    '''
    # Calculate the wave number
    k = 2 * np.pi / l

    # Calculate the polarizability using the Clausius-Mossotti relation
    alpha = (3 * epsilon_0 * (n**2 - 1)) / (n**2 + 2) * (4/3) * np.pi * (a**3)

    # Calculate the electric field amplitude of the optical trap
    E0 = np.sqrt(2 * P[0] / (np.pi * w**2 * epsilon_0 * c))

    # Calculate the dipole moment induced in each nanosphere
    p1 = alpha * E0
    p2 = alpha * E0

    # Calculate the interaction force between the dipoles
    # Using the formula for dipole-dipole interaction
    F = (1 / (4 * np.pi * epsilon_0)) * (
        (3 * p1 * p2 * np.cos(phi) / R**4) - (p1 * p2 / R**3)
    )

    return F




def generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho):
    '''Function to generate the Hamiltonian of trapped nanospheres with optical binding force appeared.
    Input
    P : list of length N
        Power of each individual optical trap.
    phi : float
        Polarization direction of the optical traps.
    R : float
        Distance between the adjacent trapped nanospheres.
    l : float
        Wavelength of the optical traps.
    w : float
        Beam waist of the optical traps.
    a : float
        Radius of the trapped microspheres.
    n : float
        Refractive index of the trapped microspheres.
    h : float
        Step size of the differentiation.
    N : int
        The total number of trapped nanospheres.
    rho: float
        Density of the trapped microspheres.
    Output
    H : matrix of shape(N, N)
        The Hamiltonian of trapped nanospheres with optical binding force appeared.
    '''

    # Wave number
    k = 2 * np.pi / l

    # Polarizability using the Clausius-Mossotti relation
    alpha = (3 * epsilon_0 * (n**2 - 1)) / (n**2 + 2) * (4/3) * np.pi * (a**3)

    # Calculate the electric field amplitude of the optical trap
    E0 = np.sqrt(2 * P[0] / (np.pi * w**2 * epsilon_0 * c))

    # Dipole moment induced in each nanosphere
    p = alpha * E0

    # Mass of the nanospheres
    mass = rho * (4/3) * np.pi * (a**3)

    # Frequency of oscillation (omega)
    omega = np.sqrt(p**2 / (mass * R**3))

    # Coupling constant (hopping strength) between adjacent particles
    coupling_constant = (p**2 / (4 * np.pi * epsilon_0 * R**3)) * np.cos(phi)

    # Initialize Hamiltonian matrix
    H = np.zeros((N, N))

    # Fill Hamiltonian matrix with coupling constants and oscillation frequencies
    for i in range(N):
        H[i, i] = omega**2  # Diagonal terms represent the frequency squared of each oscillator

        if i < N-1:
            H[i, i+1] = -coupling_constant / mass  # Off-diagonal terms represent the coupling strength
            H[i+1, i] = -coupling_constant / mass  # Symmetric matrix

    return H



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

    def lindblad_rhs(C, H, L, M):
        """Calculate the right-hand side of the Lindblad master equation."""
        # Commutator [H, C]
        comm_HC = H @ C - C @ H
        
        # Dissipative part (L*C*L^dagger - 0.5*(L^dagger*L*C + C*L^dagger*L))
        dissipation = L @ C @ L.T.conj() - 0.5 * (L.T.conj() @ L @ C + C @ L.T.conj() @ L)
        
        # Reservoir contribution
        reservoir = M
        
        # Total right-hand side
        rhs = -1j * comm_HC + dissipation + reservoir
        return rhs

    dt = t0 / steps  # Time step size
    C = C0.copy()    # Initialize the correlation matrix

    # Runge-Kutta 4th order (RK4) integration
    for _ in range(steps):
        k1 = lindblad_rhs(C, H, L, M)
        k2 = lindblad_rhs(C + 0.5 * dt * k1, H, L, M)
        k3 = lindblad_rhs(C + 0.5 * dt * k2, H, L, M)
        k4 = lindblad_rhs(C + dt * k3, H, L, M)

        C += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Phonon occupation numbers are the diagonal elements of the correlation matrix
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