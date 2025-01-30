from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import cmath
from math import pi, sin, cos, sqrt

def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
    '''Function to generate the Haldane Hamiltonian with a given set of parameters.
    Inputs:
    kx : float
        The x component of the wavevector.
    ky : float
        The y component of the wavevector.
    a : float
        The lattice spacing, i.e., the length of one side of the hexagon.
    t1 : float
        The nearest-neighbor coupling constant.
    t2 : float
        The next-nearest-neighbor coupling constant.
    phi : float
        The phase ranging from -π to π.
    m : float
        The on-site energy.
    Output:
    hamiltonian : matrix of shape(2, 2)
        The Haldane Hamiltonian on a hexagonal lattice.
    '''

    # Calculate the nearest-neighbor contributions
    d1 = np.array([a, 0])
    d2 = np.array([-a/2, sqrt(3)*a/2])
    d3 = np.array([-a/2, -sqrt(3)*a/2])

    # Calculate the phase factors for nearest-neighbor hopping
    nn_hopping = t1 * (np.exp(1j * np.dot([kx, ky], d1)) +
                       np.exp(1j * np.dot([kx, ky], d2)) +
                       np.exp(1j * np.dot([kx, ky], d3)))

    # Calculate the next-nearest-neighbor contributions
    d1_nn = np.array([3*a/2, sqrt(3)*a/2])
    d2_nn = np.array([-3*a/2, sqrt(3)*a/2])
    d3_nn = np.array([0, -sqrt(3)*a])

    # Calculate the phase factors for next-nearest-neighbor hopping
    nnn_hopping = t2 * (np.exp(1j * phi) * (np.exp(1j * np.dot([kx, ky], d1_nn)) +
                                            np.exp(1j * np.dot([kx, ky], d2_nn)) +
                                            np.exp(1j * np.dot([kx, ky], d3_nn))) +
                        np.exp(-1j * phi) * (np.exp(-1j * np.dot([kx, ky], d1_nn)) +
                                             np.exp(-1j * np.dot([kx, ky], d2_nn)) +
                                             np.exp(-1j * np.dot([kx, ky], d3_nn))))

    # Construct the Hamiltonian matrix
    hamiltonian = np.array([[m + nnn_hopping, nn_hopping],
                            [np.conj(nn_hopping), -m + nnn_hopping]])

    return hamiltonian



def compute_chern_number(delta, a, t1, t2, phi, m):
    '''Function to compute the Chern number with a given set of parameters.
    Inputs:
    delta : float
        The grid size in kx and ky axis for discretizing the Brillouin zone.
    a : float
        The lattice spacing, i.e., the length of one side of the hexagon.
    t1 : float
        The nearest-neighbor coupling constant.
    t2 : float
        The next-nearest-neighbor coupling constant.
    phi : float
        The phase ranging from -π to π.
    m : float
        The on-site energy.
    Output:
    chern_number : float
        The Chern number, a real number that should be close to an integer. The imaginary part is cropped out due to the negligible magnitude.
    '''
    
    # Define the limits of the Brillouin zone
    kx_max = 2 * np.pi / (a * sqrt(3))
    ky_max = 2 * np.pi / (3 * a)

    # Generate the k-space grid
    kx_range = np.arange(-kx_max, kx_max, delta)
    ky_range = np.arange(-ky_max, ky_max, delta)

    # Initialize the Berry curvature
    berry_curvature = 0.0

    # Loop over each point in the discretized Brillouin zone
    for kx in kx_range:
        for ky in ky_range:
            # Compute the Hamiltonian at the current k-point
            H = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
            
            # Diagonalize the Hamiltonian to obtain eigenvalues and eigenvectors
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Compute the Berry curvature at this k-point
            for n in range(len(eigvals)):
                for m in range(len(eigvals)):
                    if n != m:
                        # Gradient in k-space
                        dH_dkx = (calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m) -
                                  calc_hamiltonian(kx, ky, a, t1, t2, phi, m)) / delta
                        dH_dky = (calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m) -
                                  calc_hamiltonian(kx, ky, a, t1, t2, phi, m)) / delta
                        
                        # Berry curvature formula
                        berry_curvature += 2 * np.imag(np.vdot(eigvecs[:, n], np.dot(dH_dkx, eigvecs[:, m])) * 
                                                       np.vdot(eigvecs[:, m], np.dot(dH_dky, eigvecs[:, n]))) / \
                                           ((eigvals[n] - eigvals[m])**2)

    # Integrate the Berry curvature over the Brillouin zone
    chern_number = berry_curvature * (delta**2) / (2 * np.pi)

    # Return the real part of the Chern number
    return np.real(chern_number)


try:
    targets = process_hdf5_to_tuple('33.2', 3)
    target = targets[0]
    delta = 2 * np.pi / 200
    a = 1
    t1 = 4
    t2 = 1
    phi = 1
    m = 1
    assert np.allclose(compute_chern_number(delta, a, t1, t2, phi, m), target)

    target = targets[1]
    delta = 2 * np.pi / 100
    a = 1
    t1 = 1
    t2 = 0.3
    phi = -1
    m = 1
    assert np.allclose(compute_chern_number(delta, a, t1, t2, phi, m), target)

    target = targets[2]
    delta = 2 * np.pi / 100
    a = 1
    t1 = 1
    t2 = 0.2
    phi = 1
    m = 1
    assert np.allclose(compute_chern_number(delta, a, t1, t2, phi, m), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e