import numpy as np
import scipy
from scipy.constants import epsilon_0, c

# Background: 
# The optical binding force between two trapped nanospheres in an optical trap can be understood using the concept of induced dipoles.
# In the Rayleigh approximation, the nanospheres are much smaller than the wavelength of the trapping light, allowing them to be considered as point dipoles.
# When an electromagnetic field interacts with a dielectric sphere, it induces a dipole moment in the sphere.
# The strength of this dipole moment is dependent on the field's intensity, the sphere's volume, and its polarizability.
# The dipole moment 'p' for a sphere can be expressed as p = 4 * pi * epsilon_0 * a^3 * (n^2 - 1) / (n^2 + 2) * E,
# where 'a' is the radius of the sphere, 'n' is the refractive index, and 'E' is the electric field.
# The force 'F' between two dipoles in an electric field, separated by distance 'R', is given by:
# F = (3 * p1 * p2) / (4 * pi * epsilon_0 * R^4),
# where p1 and p2 are the magnitudes of the dipole moments of the two spheres.
# The electric field 'E' from a laser beam can be related to its power 'P' and beam waist 'w' by E = sqrt(2 * P / (pi * epsilon_0 * c * w^2)).
# This function calculates the optical binding force between two nanospheres by considering these principles.




def binding_force(P, phi, R, l, w, a, n):
    '''Function to calculate the optical binding force between two trapped nanospheres.
    Input
    P : list of length 2
        Power of the two optical traps.
    phi : float
        Polarization direction of the optical traps (not used in calculation since direction is the same).
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

    # Calculate the electric field E from the power P and beam waist w
    E1 = np.sqrt(2 * P[0] / (np.pi * epsilon_0 * c * w**2))
    E2 = np.sqrt(2 * P[1] / (np.pi * epsilon_0 * c * w**2))
    
    # Calculate the polarizability alpha of the spheres
    alpha = 4 * np.pi * epsilon_0 * a**3 * (n**2 - 1) / (n**2 + 2)
    
    # Calculate the dipole moments p1 and p2
    p1 = alpha * E1
    p2 = alpha * E2
    
    # Calculate the optical binding force F
    F = (3 * p1 * p2) / (4 * np.pi * epsilon_0 * R**4)
    
    return F



# Background: 
# In the context of optically trapped nanospheres, the optical binding force can be considered a restoring force that acts to maintain the relative positions of the spheres. When these spheres are displaced from their equilibrium positions, they can oscillate about these positions. This behavior can be modeled using coupled oscillators, where each nanosphere interacts with its neighbors through the optical binding force.
# The coupling constant, or hopping strength, between the nanoparticles is akin to the spring constant in a mechanical oscillator system, representing the strength of interaction due to optical forces.
# The Hamiltonian of such a system describes the energy, including both kinetic and potential energy, of the coupled oscillators.
# For N nanospheres, the Hamiltonian can be represented as a matrix where the diagonal elements represent the potential energy due to the optical trap, and the off-diagonal elements represent the coupling (hopping) strength between adjacent nanospheres.



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
    
    # Calculate the polarizability alpha of the spheres
    alpha = 4 * np.pi * epsilon_0 * a**3 * (n**2 - 1) / (n**2 + 2)
    
    # Calculate the electric field E from the power P and beam waist w for each trap
    E = np.sqrt(2 * np.array(P) / (np.pi * epsilon_0 * c * w**2))
    
    # Calculate the dipole moments for each sphere
    p = alpha * E
    
    # Calculate the coupling constant between adjacent spheres
    # This is the optical binding force linearized to represent the coupling strength
    k = (3 * p[:-1] * p[1:]) / (4 * np.pi * epsilon_0 * R**4)
    
    # Mass of each microsphere
    volume = (4/3) * np.pi * a**3
    mass = rho * volume
    
    # Construct the Hamiltonian matrix
    H = np.zeros((N, N), dtype=np.float64)
    
    # Diagonal elements (potential energy)
    for i in range(N):
        H[i, i] = P[i] / (h * c)  # Using a simplified model for individual trap potential energy
    
    # Off-diagonal elements (coupling terms)
    for i in range(N-1):
        H[i, i+1] = -k[i] / (mass * h**2)
        H[i+1, i] = -k[i] / (mass * h**2)
    
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
