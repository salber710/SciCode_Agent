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

    # Define the number of grid points in kx and ky directions
    num_k_points = int(2 * np.pi / delta)

    # Initialize the Berry curvature sum
    berry_curvature_sum = 0.0

    for i in range(num_k_points):
        for j in range(num_k_points):
            # Calculate the kx and ky values
            kx = i * delta
            ky = j * delta

            # Calculate the Hamiltonian at the grid point and neighboring points
            H = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
            H_kx_delta = calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m)
            H_ky_delta = calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m)
            H_kx_ky_delta = calc_hamiltonian(kx + delta, ky + delta, a, t1, t2, phi, m)

            # Diagonalize the Hamiltonian to get the eigenvectors
            _, eigvecs = np.linalg.eigh(H)
            _, eigvecs_kx_delta = np.linalg.eigh(H_kx_delta)
            _, eigvecs_ky_delta = np.linalg.eigh(H_ky_delta)
            _, eigvecs_kx_ky_delta = np.linalg.eigh(H_kx_ky_delta)

            # Extract the eigenvectors for the occupied band (index 0 for lowest band)
            u = eigvecs[:, 0]
            u_kx_delta = eigvecs_kx_delta[:, 0]
            u_ky_delta = eigvecs_ky_delta[:, 0]
            u_kx_ky_delta = eigvecs_kx_ky_delta[:, 0]

            # Calculate the Berry phase around the plaquette using the Wilson loop method
            # Note: np.vdot computes the complex conjugate of the first vector
            berry_phase = np.angle(
                np.vdot(u, u_kx_delta) *
                np.vdot(u_kx_delta, u_kx_ky_delta) *
                np.vdot(u_kx_ky_delta, u_ky_delta) *
                np.vdot(u_ky_delta, u)
            )

            # Accumulate the Berry curvature
            berry_curvature_sum += berry_phase

    # The Chern number is the integral of the Berry curvature over the Brillouin zone
    # divided by 2*pi
    chern_number = berry_curvature_sum / (2 * np.pi)

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