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
    
    return F

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('32.1', 3)
target = targets[0]

P = [10000000, 100000000]
phi = 0
R = 1550e-9
l = 1550e-9
w = 600e-9
a = 100e-9
n = 1.444
assert np.allclose(binding_force(P, phi, R, l, w, a, n), target)
target = targets[1]

P = [10000000, 100000000]
phi = np.pi/2
R = 1550e-9
l = 1550e-9
w = 600e-9
a = 100e-9
n = 1.444
assert np.allclose(binding_force(P, phi, R, l, w, a, n), target)
target = targets[2]

P = [1000000000, 1000000000]
phi = np.pi/4
R = 1550e-9
l = 1550e-9
w = 600e-9
a = 100e-9
n = 1.444
assert np.allclose(binding_force(P, phi, R, l, w, a, n), target)
