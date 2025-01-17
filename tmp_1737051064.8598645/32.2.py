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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('32.2', 3)
target = targets[0]

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
assert np.allclose(generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho), target)
target = targets[1]

P = [100e-3, 100e-3, 100e-3, 100e-3, 100e-3]
phi = np.pi / 2
R = 2 * 1550e-9
l = 1550e-9
w = 600e-9
a = 100e-9
n = 1.444
h = 1e-6
N = np.size(P)
rho = 2.648e3
assert np.allclose(generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho), target)
target = targets[2]

P = [100e-3, 100e-3, 100e-3, 100e-3, 100e-3]
phi = 0
R = 1 * 1550e-9
l = 1550e-9
w = 600e-9
a = 100e-9
n = 1.444
h = 1e-6
N = np.size(P)
rho = 2.648e3
assert np.allclose(generate_Hamiltonian(P, phi, R, l, w, a, n, h, N, rho), target)
