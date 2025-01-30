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
    # Hamiltonian matrix
    H = np.zeros((N, N))
    
    # Calculate the wave number
    k = 2 * np.pi / l

    # Calculate the polarizability using the Clausius-Mossotti relation
    alpha = (3 * epsilon_0 * (n**2 - 1)) / (n**2 + 2) * (4/3) * np.pi * (a**3)

    # Calculate the electric field amplitude of the optical trap
    E0 = np.sqrt(2 * P[0] / (np.pi * w**2 * epsilon_0 * c))

    # Calculate the dipole moment induced in each nanosphere
    p = alpha * E0

    # Calculate the coupling constant (hopping strength) between nanoparticles
    # Using the formula for dipole-dipole interaction
    coupling_constant = (1 / (4 * np.pi * epsilon_0)) * (
        (3 * p**2 * np.cos(phi) / R**4) - (p**2 / R**3)
    )

    # Fill the Hamiltonian matrix with coupling constants
    for i in range(N - 1):
        H[i, i+1] = coupling_constant
        H[i+1, i] = coupling_constant  # Hamiltonian is Hermitian

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