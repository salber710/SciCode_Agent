import numpy as np
import scipy
from scipy.constants import epsilon_0, c

# Background: In the Rayleigh approximation, nanospheres are considered as induced dipoles in an external electric field.
# The optical binding force between two such dipoles can be calculated by considering the interaction between the induced
# dipole moment of one nanosphere and the electric field produced by the other. The dipole moment p of a sphere in an
# electric field E is given by p = αE, where α is the polarizability of the sphere. The polarizability α can be calculated
# using the formula α = 4πε₀a³((n²-1)/(n²+2)), where a is the radius of the sphere, n is the refractive index, and ε₀ is
# the permittivity of free space. The electric field E produced by a dipole at a distance R is given by E = (1/(4πε₀)) * 
# (3(n·p)n - p) / R³, where n is the unit vector in the direction of R. The force F on a dipole p in an electric field E
# is given by F = (p·∇)E. In this context, the optical binding force can be derived from these principles.



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
    
    # Check for negative power values
    if P[0] < 0 or P[1] < 0:
        raise ValueError("Power values must be non-negative.")
    
    # Check for negative or zero radius
    if a <= 0:
        raise ValueError("Radius must be positive and non-zero.")
    
    # Calculate the polarizability of the nanospheres
    alpha = 4 * np.pi * epsilon_0 * a**3 * ((n**2 - 1) / (n**2 + 2))
    
    # Calculate the intensity of the optical traps
    I1 = 2 * P[0] / (np.pi * w**2)
    I2 = 2 * P[1] / (np.pi * w**2)
    
    # Calculate the electric field amplitudes
    E1 = np.sqrt(2 * I1 / (c * epsilon_0))
    E2 = np.sqrt(2 * I2 / (c * epsilon_0))
    
    # Calculate the dipole moments
    p1 = alpha * E1
    p2 = alpha * E2
    
    # Calculate the unit vector in the direction of R
    n_vec = np.array([np.cos(phi), np.sin(phi), 0])
    
    # Calculate the electric field at the position of the second dipole due to the first dipole
    E_at_2 = (1 / (4 * np.pi * epsilon_0)) * ((3 * np.dot(n_vec, p1) * n_vec - p1) / R**3)
    
    # Calculate the force on the second dipole due to the electric field from the first dipole
    F = np.dot(p2, E_at_2)
    
    # Return the magnitude of the force vector
    return np.linalg.norm(F)



# Background: In the context of optical binding forces and small vibrations around equilibrium positions, the system of
# trapped nanospheres can be modeled as coupled oscillators. The coupling constant, or hopping strength, between these
# oscillators is related to the optical binding force. The Hamiltonian of such a system describes the energy and dynamics
# of the coupled oscillators. For a system of N nanospheres, the Hamiltonian is an N x N matrix where the diagonal elements
# represent the self-energy of each nanosphere, and the off-diagonal elements represent the coupling between adjacent
# nanospheres. The coupling constant can be derived from the linearized optical binding force, which is proportional to
# the derivative of the force with respect to the displacement from equilibrium.



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
    
    # Calculate the dipole moments for each nanosphere
    p = [alpha * E[i] for i in range(N)]
    
    # Calculate the unit vector in the direction of R
    n_vec = np.array([np.cos(phi), np.sin(phi), 0])
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((N, N))
    
    # Calculate the coupling constant (hopping strength) between adjacent nanospheres
    for i in range(N - 1):
        # Calculate the electric field at the position of the (i+1)th dipole due to the ith dipole
        E_at_next = (1 / (4 * np.pi * epsilon_0)) * ((3 * np.dot(n_vec, p[i]) * n_vec - p[i]) / R**3)
        
        # Calculate the force on the (i+1)th dipole due to the electric field from the ith dipole
        F = np.dot(p[i+1], E_at_next)
        
        # Linearize the force to find the coupling constant
        coupling_constant = F / h
        
        # Fill the Hamiltonian matrix
        H[i, i] = 2 * coupling_constant  # Self-energy term
        H[i, i+1] = -coupling_constant  # Coupling term
        H[i+1, i] = -coupling_constant  # Coupling term
    
    # Return the Hamiltonian matrix
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
