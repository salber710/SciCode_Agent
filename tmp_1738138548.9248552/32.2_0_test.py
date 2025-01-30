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



# Background: In the context of optically trapped nanospheres, the optical binding force can be linearized
# when considering small vibrations around equilibrium positions. This allows the system to be modeled as
# a set of coupled oscillators. The coupling constant, or hopping strength, between these oscillators
# (nanospheres) is determined by the optical binding force. The Hamiltonian of such a system describes
# the energy and interactions between the oscillators. In this case, the Hamiltonian is a matrix where
# the diagonal elements represent the self-energy of each nanosphere, and the off-diagonal elements
# represent the coupling between adjacent nanospheres due to the optical binding force.



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
    
    # Calculate polarizability
    alpha = 4 * np.pi * epsilon_0 * a**3 * ((n**2 - 1) / (n**2 + 2))
    
    # Calculate the electric field amplitude for each trap
    E0 = np.array([np.sqrt(2 * P[i] / (np.pi * w**2 * epsilon_0)) for i in range(N)])
    
    # Calculate the dipole moment induced in each sphere
    p = alpha * E0
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((N, N), dtype=np.complex128)
    
    # Calculate the coupling constant (hopping strength) between adjacent nanospheres
    for i in range(N - 1):
        # Calculate the electric field at the location of the (i+1)th sphere due to the ith sphere
        r_hat = np.array([np.cos(phi), np.sin(phi), 0])
        E_field = (1 / (4 * np.pi * epsilon_0)) * (3 * np.dot(p[i], r_hat) * r_hat - p[i]) / R**3
        
        # Calculate the force on the (i+1)th dipole
        F = np.dot(p[i+1], E_field)
        
        # The coupling constant is related to the force and the displacement
        coupling_constant = F * h
        
        # Fill the Hamiltonian matrix
        H[i, i+1] = coupling_constant
        H[i+1, i] = coupling_constant
    
    # The diagonal elements can be set to some self-energy term, here assumed to be zero for simplicity
    # This can be modified based on the specific physical model
    for i in range(N):
        H[i, i] = 0  # or some self-energy term if needed
    
    return H


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e