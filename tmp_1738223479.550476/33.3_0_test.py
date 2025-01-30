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

    # Define the hexagonal lattice vectors using an unconventional method
    nn_angles = np.array([0, 2 * np.pi / 3, -2 * np.pi / 3])
    nnn_angles = np.array([np.pi / 3, -np.pi / 3, np.pi, -np.pi, 5 * np.pi / 3, -5 * np.pi / 3])

    # Nearest-neighbor contributions using polar form
    f_k = t1 * np.sum(np.exp(1j * (kx * a * np.cos(nn_angles) + ky * a * np.sin(nn_angles))))

    # Next-nearest-neighbor contributions with phase
    g_k = t2 * np.sum(np.exp(1j * (kx * a * np.cos(nnn_angles) + ky * a * np.sin(nnn_angles) + phi)))

    # Construct the Haldane Hamiltonian matrix in a unique structure
    hamiltonian = np.array([[m + g_k, f_k],
                            [np.conj(f_k), -m + np.conj(g_k)]])
    
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
        The Chern number, a real number that should be close to an integer.
    '''

    def haldane_hamiltonian(kx, ky):
        # Alternative Hamiltonian construction using symmetry considerations
        h_0 = m
        h_x = -t1 * (np.cos(kx * a) + np.cos(ky * a) + np.cos((kx + ky) * a))
        h_y = -t1 * (np.sin(kx * a) + np.sin(ky * a) + np.sin((kx + ky) * a))
        h_z = -2 * t2 * np.sin(phi) * (np.sin(kx * a) + np.sin(ky * a) + np.sin((kx + ky) * a))
        
        return np.array([[h_0 + h_z, h_x - 1j * h_y], [h_x + 1j * h_y, h_0 - h_z]])
    
    def berry_curvature(kx, ky):
        hamiltonian = haldane_hamiltonian(kx, ky)
        eigvals, eigvecs = np.linalg.eigh(hamiltonian)
        occupied = eigvecs[:, 0]

        # Use central difference method for better accuracy in finite differences
        dk = delta
        h_dkx = haldane_hamiltonian(kx + dk, ky)
        h_dky = haldane_hamiltonian(kx, ky + dk)

        _, vec_dkx = np.linalg.eigh(h_dkx)
        _, vec_dky = np.linalg.eigh(h_dky)

        u_dkx = vec_dkx[:, 0]
        u_dky = vec_dky[:, 0]

        # Calculate Berry phase using a different path
        phase_factor = np.angle(
            np.vdot(occupied, u_dkx) *
            np.vdot(u_dkx, u_dky) *
            np.vdot(u_dky, occupied)
        )
        
        return phase_factor / (dk * dk)

    # Set up k-space grid
    kx_vals = np.arange(-np.pi/a, np.pi/a, delta)
    ky_vals = np.arange(-np.pi/a, np.pi/a, delta)

    chern_number = 0.0

    # Perform integration over the Brillouin zone using a different summation order
    for ky in ky_vals:
        for kx in kx_vals:
            chern_number += berry_curvature(kx, ky)

    # Normalize the result
    chern_number *= (delta**2 / (2 * np.pi))

    return chern_number



# Background: The Chern number is a topological invariant used to characterize phases of matter in condensed matter physics, particularly in systems exhibiting the quantum Hall effect. It is computed by integrating the Berry curvature over the Brillouin zone, which provides a quantized value indicating topological order. In this context, we aim to map out how the Chern number changes as a function of the ratio of on-site energy to next-nearest-neighbor coupling (m/t2) and the phase (phi) in the Haldane model. This involves sweeping these parameters over specified ranges and calculating the Chern number for each combination, providing a 2D array of these values.

def compute_chern_number_grid(delta, a, t1, t2, N):
    '''Function to calculate the Chern numbers by sweeping the given set of parameters and returns the results along with the corresponding swept next-nearest-neighbor coupling constant and phase.
    Inputs:
    delta : float
        The grid size in kx and ky axis for discretizing the Brillouin zone.
    a : float
        The lattice spacing, i.e., the length of one side of the hexagon.
    t1 : float
        The nearest-neighbor coupling constant.
    t2 : float
        The next-nearest-neighbor coupling constant.
    N : int
        The number of sweeping grid points for both the on-site energy to next-nearest-neighbor coupling constant ratio and phase.
    Outputs:
    results: matrix of shape(N, N)
        The Chern numbers by sweeping the on-site energy to next-nearest-neighbor coupling constant ratio (m/t2) and phase (phi).
    m_values: array of length N
        The swept on-site energy to next-nearest-neighbor coupling constant ratios.
    phi_values: array of length N
        The swept phase values.
    '''

    m_values = np.linspace(-6 * t2, 6 * t2, N)
    phi_values = np.linspace(-pi, pi, N)
    results = np.zeros((N, N))

    def compute_chern_number(delta, a, t1, t2, phi, m):
        # Redefine the Hamiltonian and Berry curvature calculations within this scope
        def haldane_hamiltonian(kx, ky):
            h_0 = m
            h_x = -t1 * (cos(kx * a) + cos(ky * a) + cos((kx + ky) * a))
            h_y = -t1 * (sin(kx * a) + sin(ky * a) + sin((kx + ky) * a))
            h_z = -2 * t2 * sin(phi) * (sin(kx * a) + sin(ky * a) + sin((kx + ky) * a))
            return np.array([[h_0 + h_z, h_x - 1j * h_y], [h_x + 1j * h_y, h_0 - h_z]])

        def berry_curvature(kx, ky):
            hamiltonian = haldane_hamiltonian(kx, ky)
            eigvals, eigvecs = np.linalg.eigh(hamiltonian)
            occupied = eigvecs[:, 0]

            dk = delta
            h_dkx = haldane_hamiltonian(kx + dk, ky)
            h_dky = haldane_hamiltonian(kx, ky + dk)

            _, vec_dkx = np.linalg.eigh(h_dkx)
            _, vec_dky = np.linalg.eigh(h_dky)

            u_dkx = vec_dkx[:, 0]
            u_dky = vec_dky[:, 0]

            phase_factor = np.angle(
                np.vdot(occupied, u_dkx) *
                np.vdot(u_dkx, u_dky) *
                np.vdot(u_dky, occupied)
            )
            return phase_factor / (dk * dk)

        kx_vals = np.arange(-pi/a, pi/a, delta)
        ky_vals = np.arange(-pi/a, pi/a, delta)

        chern_number = 0.0
        for ky in ky_vals:
            for kx in kx_vals:
                chern_number += berry_curvature(kx, ky)

        chern_number *= (delta**2 / (2 * pi))
        return chern_number

    for i, m in enumerate(m_values):
        for j, phi in enumerate(phi_values):
            results[i, j] = compute_chern_number(delta, a, t1, t2, phi, m)

    return results, m_values, phi_values


try:
    targets = process_hdf5_to_tuple('33.3', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    delta = 2 * np.pi / 30
    a = 1.0
    t1 = 4.0
    t2 = 1.0
    N = 40
    assert cmp_tuple_or_list(compute_chern_number_grid(delta, a, t1, t2, N), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    delta = 2 * np.pi / 30
    a = 1.0
    t1 = 5.0
    t2 = 1.0
    N = 40
    assert cmp_tuple_or_list(compute_chern_number_grid(delta, a, t1, t2, N), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    delta = 2 * np.pi / 30
    a = 1.0
    t1 = 1.0
    t2 = 0.2
    N = 40
    assert cmp_tuple_or_list(compute_chern_number_grid(delta, a, t1, t2, N), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e