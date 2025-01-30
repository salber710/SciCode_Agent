from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import cmath
from math import pi, sin, cos, sqrt


def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
    # Define the primitive vectors for the hexagonal lattice
    v1 = np.array([np.sqrt(3)/2, 0.5]) * a
    v2 = np.array([-np.sqrt(3)/2, 0.5]) * a

    # Nearest-neighbor vectors
    deltas = [v1, v2, -v1 - v2]

    # Next-nearest-neighbor vectors
    betas = [v1 - v2, v2 - v1, 2*v1, -2*v1, 2*v2, -2*v2]

    # Calculate the nearest-neighbor hopping term
    f_k = t1 * sum(np.exp(1j * (kx * delta[0] + ky * delta[1])) for delta in deltas)

    # Calculate the next-nearest-neighbor hopping term with phase
    g_k = t2 * sum(np.exp(1j * (kx * beta[0] + ky * beta[1] + phi)) for beta in betas)

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([
        [m + np.real(g_k), f_k],
        [np.conj(f_k), -m + np.real(g_k)]
    ])

    return hamiltonian



# Background: The Chern number is a topological invariant that characterizes the global properties of a band structure in a periodic system. 
# For a two-dimensional system like the Haldane model, the Chern number can be computed by integrating the Berry curvature over the Brillouin zone.
# The Berry curvature is derived from the eigenstates of the Hamiltonian. For a discretized Brillouin zone, the Chern number can be approximated 
# using a finite difference method. The Chern number is related to the quantized Hall conductance in the quantum Hall effect.




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
    
    def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
        # Define the primitive vectors for the hexagonal lattice
        v1 = np.array([np.sqrt(3)/2, 0.5]) * a
        v2 = np.array([-np.sqrt(3)/2, 0.5]) * a

        # Nearest-neighbor vectors
        deltas = [v1, v2, -v1 - v2]

        # Next-nearest-neighbor vectors
        betas = [v1 - v2, v2 - v1, 2*v1, -2*v1, 2*v2, -2*v2]

        # Calculate the nearest-neighbor hopping term
        f_k = t1 * sum(np.exp(1j * (kx * delta[0] + ky * delta[1])) for delta in deltas)

        # Calculate the next-nearest-neighbor hopping term with phase
        g_k = t2 * sum(np.exp(1j * (kx * beta[0] + ky * beta[1] + phi)) for beta in betas)

        # Construct the Haldane Hamiltonian matrix
        hamiltonian = np.array([
            [m + np.real(g_k), f_k],
            [np.conj(f_k), -m + np.real(g_k)]
        ])

        return hamiltonian

    # Discretize the Brillouin zone
    kx_vals = np.arange(-pi/a, pi/a, delta)
    ky_vals = np.arange(-pi/a, pi/a, delta)

    # Initialize the Chern number
    chern_number = 0.0

    # Loop over the discretized Brillouin zone
    for i in range(len(kx_vals) - 1):
        for j in range(len(ky_vals) - 1):
            # Calculate the Hamiltonians at the corners of the plaquette
            H00 = calc_hamiltonian(kx_vals[i], ky_vals[j], a, t1, t2, phi, m)
            H10 = calc_hamiltonian(kx_vals[i+1], ky_vals[j], a, t1, t2, phi, m)
            H01 = calc_hamiltonian(kx_vals[i], ky_vals[j+1], a, t1, t2, phi, m)
            H11 = calc_hamiltonian(kx_vals[i+1], ky_vals[j+1], a, t1, t2, phi, m)

            # Calculate the eigenvectors
            _, v00 = np.linalg.eigh(H00)
            _, v10 = np.linalg.eigh(H10)
            _, v01 = np.linalg.eigh(H01)
            _, v11 = np.linalg.eigh(H11)

            # Calculate the Berry phase for the plaquette
            U1 = np.vdot(v00[:, 0], v10[:, 0])
            U2 = np.vdot(v10[:, 0], v11[:, 0])
            U3 = np.vdot(v11[:, 0], v01[:, 0])
            U4 = np.vdot(v01[:, 0], v00[:, 0])

            # Calculate the Berry curvature for the plaquette
            F_ij = cmath.log(U1 * U2 * U3 * U4).imag

            # Sum the Berry curvature to get the Chern number
            chern_number += F_ij

    # Normalize the Chern number
    chern_number /= (2 * pi)

    return chern_number


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