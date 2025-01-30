import numpy as np
import cmath
from math import pi, sin, cos, sqrt

# Background: The Haldane model is a theoretical model used to describe a two-dimensional electron system on a hexagonal lattice, such as graphene. 
# It includes both nearest-neighbor and next-nearest-neighbor interactions. The Hamiltonian for the Haldane model is a 2x2 matrix that describes 
# the energy of the system in terms of the wavevector components (kx, ky), lattice spacing (a), nearest-neighbor coupling (t1), 
# next-nearest-neighbor coupling (t2), a phase (phi) associated with the next-nearest-neighbor hopping, and an on-site energy (m).
# The nearest-neighbor hopping terms contribute to the off-diagonal elements of the Hamiltonian, while the next-nearest-neighbor terms 
# and the on-site energy contribute to the diagonal elements. The phase phi introduces a complex phase factor in the next-nearest-neighbor 
# hopping, which can lead to a non-trivial topological phase.




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

    # Define the nearest-neighbor vectors for a hexagonal lattice
    delta1 = np.array([1, 0]) * a
    delta2 = np.array([-0.5, sqrt(3)/2]) * a
    delta3 = np.array([-0.5, -sqrt(3)/2]) * a

    # Calculate the nearest-neighbor hopping terms
    gamma1 = t1 * (cmath.exp(1j * np.dot(delta1, [kx, ky])) +
                   cmath.exp(1j * np.dot(delta2, [kx, ky])) +
                   cmath.exp(1j * np.dot(delta3, [kx, ky])))

    # Define the next-nearest-neighbor vectors
    eta1 = np.array([1.5, sqrt(3)/2]) * a
    eta2 = np.array([0, -sqrt(3)]) * a
    eta3 = np.array([-1.5, sqrt(3)/2]) * a

    # Calculate the next-nearest-neighbor hopping terms
    gamma2 = t2 * (cmath.exp(1j * (np.dot(eta1, [kx, ky]) + phi)) +
                   cmath.exp(1j * (np.dot(eta2, [kx, ky]) + phi)) +
                   cmath.exp(1j * (np.dot(eta3, [kx, ky]) + phi)))

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([[m + 2 * gamma2.real, gamma1],
                            [np.conj(gamma1), -m + 2 * gamma2.real]])

    return hamiltonian


# Background: The Chern number is a topological invariant that characterizes the global properties of a band structure in a periodic system. 
# In the context of the Haldane model, it is used to determine the topological phase of the system. The Chern number is calculated by integrating 
# the Berry curvature over the Brillouin zone. For a discretized Brillouin zone, this can be approximated by summing the Berry curvature over 
# a grid of points. The Berry curvature can be computed using the eigenvectors of the Hamiltonian at each point in the Brillouin zone.




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

    if delta <= 0 or a <= 0:
        raise ValueError("Delta and lattice spacing 'a' must be positive numbers.")

    def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
        delta1 = np.array([1, 0]) * a
        delta2 = np.array([-0.5, sqrt(3)/2]) * a
        delta3 = np.array([-0.5, -sqrt(3)/2]) * a

        gamma1 = t1 * (cmath.exp(1j * np.dot(delta1, [kx, ky])) +
                       cmath.exp(1j * np.dot(delta2, [kx, ky])) +
                       cmath.exp(1j * np.dot(delta3, [kx, ky])))

        eta1 = np.array([1.5, sqrt(3)/2]) * a
        eta2 = np.array([0, -sqrt(3)]) * a
        eta3 = np.array([-1.5, sqrt(3)/2]) * a

        gamma2 = t2 * (cmath.exp(1j * (np.dot(eta1, [kx, ky]) + phi)) +
                       cmath.exp(1j * (np.dot(eta2, [kx, ky]) + phi)) +
                       cmath.exp(1j * (np.dot(eta3, [kx, ky]) + phi)))

        hamiltonian = np.array([[m + 2 * gamma2.real, gamma1],
                                [np.conj(gamma1), -m + 2 * gamma2.real]])

        return hamiltonian

    def berry_curvature(kx, ky, a, t1, t2, phi, m):
        h = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
        eigvals, eigvecs = np.linalg.eigh(h)
        u = eigvecs[:, 0]  # Ground state eigenvector
        u_kx = (calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m) - h) @ u / delta
        u_ky = (calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m) - h) @ u / delta
        f_xy = 1j * (np.vdot(u, u_kx) * np.vdot(u_ky, u) - np.vdot(u, u_ky) * np.vdot(u_kx, u))
        return f_xy

    kx_vals = np.arange(-pi/a, pi/a, delta)
    ky_vals = np.arange(-pi/a, pi/a, delta)
    chern_number = 0.0

    for kx in kx_vals:
        for ky in ky_vals:
            chern_number += berry_curvature(kx, ky, a, t1, t2, phi, m).imag

    chern_number *= (delta**2) / (2 * pi)
    return chern_number



# Background: The Chern number is a topological invariant that characterizes the global properties of a band structure in a periodic system. 
# In the context of the Haldane model, it is used to determine the topological phase of the system. By sweeping parameters such as the on-site 
# energy to next-nearest-neighbor coupling ratio (m/t2) and the phase (phi), we can explore how the Chern number changes across different 
# parameter regimes. This is useful for mapping out the phase diagram of the system. The Chern number is calculated by integrating the Berry 
# curvature over the Brillouin zone, which can be approximated by summing the Berry curvature over a grid of points in a discretized Brillouin 
# zone. The Berry curvature is computed using the eigenvectors of the Hamiltonian at each point in the Brillouin zone.




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

    def calc_hamiltonian(kx, ky, a, t1, t2, phi, m):
        delta1 = np.array([1, 0]) * a
        delta2 = np.array([-0.5, sqrt(3)/2]) * a
        delta3 = np.array([-0.5, -sqrt(3)/2]) * a

        gamma1 = t1 * (cmath.exp(1j * np.dot(delta1, [kx, ky])) +
                       cmath.exp(1j * np.dot(delta2, [kx, ky])) +
                       cmath.exp(1j * np.dot(delta3, [kx, ky])))

        eta1 = np.array([1.5, sqrt(3)/2]) * a
        eta2 = np.array([0, -sqrt(3)]) * a
        eta3 = np.array([-1.5, sqrt(3)/2]) * a

        gamma2 = t2 * (cmath.exp(1j * (np.dot(eta1, [kx, ky]) + phi)) +
                       cmath.exp(1j * (np.dot(eta2, [kx, ky]) + phi)) +
                       cmath.exp(1j * (np.dot(eta3, [kx, ky]) + phi)))

        hamiltonian = np.array([[m + 2 * gamma2.real, gamma1],
                                [np.conj(gamma1), -m + 2 * gamma2.real]])

        return hamiltonian

    def berry_curvature(kx, ky, a, t1, t2, phi, m):
        h = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
        eigvals, eigvecs = np.linalg.eigh(h)
        u = eigvecs[:, 0]  # Ground state eigenvector
        u_kx = (calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m) - h) @ u / delta
        u_ky = (calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m) - h) @ u / delta
        f_xy = 1j * (np.vdot(u, u_kx) * np.vdot(u_ky, u) - np.vdot(u, u_ky) * np.vdot(u_kx, u))
        return f_xy

    m_values = np.linspace(-6 * t2, 6 * t2, N)
    phi_values = np.linspace(-pi, pi, N)
    results = np.zeros((N, N))

    for i, m in enumerate(m_values):
        for j, phi in enumerate(phi_values):
            kx_vals = np.arange(-pi/a, pi/a, delta)
            ky_vals = np.arange(-pi/a, pi/a, delta)
            chern_number = 0.0

            for kx in kx_vals:
                for ky in ky_vals:
                    chern_number += berry_curvature(kx, ky, a, t1, t2, phi, m).imag

            chern_number *= (delta**2) / (2 * pi)
            results[i, j] = chern_number

    return results, m_values, phi_values

from scicode.parse.parse import process_hdf5_to_tuple
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
