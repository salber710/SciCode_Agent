import numpy as np
import scipy
from scipy.constants import epsilon_0, c

# Background: 
# In the Rayleigh approximation, nanospheres are considered as induced dipoles in an external electric field. 
# The optical binding force between two such dipoles can be calculated by considering the interaction between 
# the induced dipole of one nanosphere and the electric field produced by the other. The induced dipole moment 
# p of a sphere in an electric field E is given by p = αE, where α is the polarizability of the sphere. 
# The polarizability α for a sphere of radius a and refractive index n in a medium with permittivity ε_0 is 
# given by α = 4πε_0a^3((n^2 - 1)/(n^2 + 2)). The electric field E produced by a dipole at a distance R is 
# E = (1/(4πε_0)) * (3(n·p)n - p) / R^3, where n is the unit vector in the direction of R. The force F on a 
# dipole p in an electric field E is F = (p·∇)E. For simplicity, we assume the polarization direction is along 
# the x-axis and the distance R is along the z-axis.



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
    
    # Calculate the polarizability of the nanospheres
    alpha = 4 * np.pi * epsilon_0 * a**3 * ((n**2 - 1) / (n**2 + 2))
    
    # Calculate the intensity of the optical traps
    I1 = 2 * P[0] / (np.pi * w**2)
    I2 = 2 * P[1] / (np.pi * w**2)
    
    # Calculate the electric field amplitudes
    E1 = np.sqrt(2 * I1 / (c * epsilon_0))
    E2 = np.sqrt(2 * I2 / (c * epsilon_0))
    
    # Induced dipole moments
    p1 = alpha * E1
    p2 = alpha * E2
    
    # Unit vector in the direction of R (assuming R is along the z-axis)
    n_vec = np.array([0, 0, 1])
    
    # Electric field at the position of the second dipole due to the first dipole
    E_at_2 = (1 / (4 * np.pi * epsilon_0)) * (3 * np.dot(n_vec, p1) * n_vec - p1) / R**3
    
    # Optical binding force on the second dipole due to the first dipole
    F = np.dot(p2, E_at_2)
    
    return F


# Background: 
# In the context of optical binding forces and small vibrations around equilibrium positions, the system of 
# nanoparticles can be modeled as coupled oscillators. The coupling constant, or hopping strength, between 
# nanoparticles is related to the optical binding force and the displacement from equilibrium. The Hamiltonian 
# of such a system describes the energy and interactions between these oscillators. For a system of N 
# nanoparticles, the Hamiltonian is an N x N matrix where the diagonal elements represent the self-energy of 
# each nanoparticle, and the off-diagonal elements represent the coupling between adjacent nanoparticles. 
# The coupling constant can be derived from the linearized optical binding force, which is proportional to 
# the derivative of the force with respect to displacement. The density of the microspheres and the step size 
# for differentiation are used to calculate the mass and the coupling constant.



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
    
    # Calculate the polarizability of the nanospheres
    alpha = 4 * np.pi * epsilon_0 * a**3 * ((n**2 - 1) / (n**2 + 2))
    
    # Calculate the intensity and electric field amplitudes for each trap
    I = [2 * P[i] / (np.pi * w**2) for i in range(N)]
    E = [np.sqrt(2 * I[i] / (c * epsilon_0)) for i in range(N)]
    
    # Induced dipole moments for each nanosphere
    p = [alpha * E[i] for i in range(N)]
    
    # Mass of each nanosphere
    volume = (4/3) * np.pi * a**3
    mass = rho * volume
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((N, N))
    
    # Calculate the coupling constant (hopping strength) between adjacent nanospheres
    for i in range(N - 1):
        # Electric field at the position of the i+1-th dipole due to the i-th dipole
        n_vec = np.array([0, 0, 1])  # Unit vector along the z-axis
        E_at_next = (1 / (4 * np.pi * epsilon_0)) * (3 * np.dot(n_vec, p[i]) * n_vec - p[i]) / R**3
        
        # Optical binding force on the i+1-th dipole due to the i-th dipole
        F = np.dot(p[i+1], E_at_next)
        
        # Linearize the force to find the coupling constant
        # F' = dF/dR, approximated by (F(R+h) - F(R-h)) / (2*h)
        E_at_next_plus = (1 / (4 * np.pi * epsilon_0)) * (3 * np.dot(n_vec, p[i]) * n_vec - p[i]) / (R + h)**3
        F_plus = np.dot(p[i+1], E_at_next_plus)
        
        E_at_next_minus = (1 / (4 * np.pi * epsilon_0)) * (3 * np.dot(n_vec, p[i]) * n_vec - p[i]) / (R - h)**3
        F_minus = np.dot(p[i+1], E_at_next_minus)
        
        F_prime = (F_plus - F_minus) / (2 * h)
        
        # Coupling constant (hopping strength)
        coupling_constant = F_prime / mass
        
        # Fill the Hamiltonian matrix
        H[i, i] += coupling_constant
        H[i+1, i+1] += coupling_constant
        H[i, i+1] -= coupling_constant
        H[i+1, i] -= coupling_constant
    
    return H



# Background: 
# The Runge-Kutta method is a numerical technique used to solve ordinary differential equations (ODEs). 
# The fourth-order Runge-Kutta (RK4) method is particularly popular due to its balance between accuracy and 
# computational efficiency. In the context of quantum mechanics, the Lindblad master equation describes the 
# time evolution of the density matrix of an open quantum system, accounting for both the unitary evolution 
# due to the system Hamiltonian and the non-unitary evolution due to dissipation and interaction with a 
# reservoir. The correlation matrix C_ij = ⟨b_i^† b_j⟩ represents the expectation values of the phonon 
# occupation numbers, and its dynamics can be described by the Lindblad equation. The RK4 method can be 
# applied to integrate this equation over time, providing a numerical solution for the phonon occupation.



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
    
    # Function to compute the derivative of the correlation matrix
    def dCdt(C, H, L, M):
        # Commutator term: -i[H, C]
        commutator = -1j * (H @ C - C @ H)
        
        # Dissipation term: L @ C @ L^† - 0.5 * {L^†L, C}
        dissipation = L @ C @ L.conj().T - 0.5 * (L.conj().T @ L @ C + C @ L.conj().T @ L)
        
        # Reservoir term: M
        reservoir = M
        
        # Total derivative
        return commutator + dissipation + reservoir
    
    # Initialize the correlation matrix
    C = C0
    
    # Runge-Kutta 4th order integration
    for _ in range(steps):
        k1 = dCdt(C, H, L, M)
        k2 = dCdt(C + 0.5 * dt * k1, H, L, M)
        k3 = dCdt(C + 0.5 * dt * k2, H, L, M)
        k4 = dCdt(C + dt * k3, H, L, M)
        
        # Update the correlation matrix
        C += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Calculate the phonon occupation numbers (diagonal elements of the correlation matrix)
    nf = np.diag(C).real
    
    return nf


from scicode.parse.parse import process_hdf5_to_tuple

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
