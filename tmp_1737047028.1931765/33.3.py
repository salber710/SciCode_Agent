import numpy as np
import cmath
from math import pi, sin, cos, sqrt

# Background: The Haldane model is a theoretical model used to describe a two-dimensional electron system on a hexagonal lattice, such as graphene. 
# It includes both nearest-neighbor and next-nearest-neighbor interactions. The Hamiltonian for the Haldane model is a 2x2 matrix that describes 
# the energy of the system in terms of the wavevector components (kx, ky), lattice spacing (a), nearest-neighbor coupling (t1), 
# next-nearest-neighbor coupling (t2), a phase (phi) associated with the next-nearest-neighbor hopping, and an on-site energy (m).
# The nearest-neighbor hopping terms contribute to the off-diagonal elements of the Hamiltonian, while the next-nearest-neighbor terms 
# and the on-site energy contribute to the diagonal elements. The phase phi introduces a complex phase to the next-nearest-neighbor hopping, 
# which can lead to topologically non-trivial phases.




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
    t1_term = t1 * (np.exp(1j * np.dot([kx, ky], delta1)) +
                    np.exp(1j * np.dot([kx, ky], delta2)) +
                    np.exp(1j * np.dot([kx, ky], delta3)))

    # Define the next-nearest-neighbor vectors
    delta1_prime = delta1 + delta2
    delta2_prime = delta2 + delta3
    delta3_prime = delta3 + delta1

    # Calculate the next-nearest-neighbor hopping terms with phase
    t2_term = t2 * (np.exp(1j * (np.dot([kx, ky], delta1_prime) + phi)) +
                    np.exp(1j * (np.dot([kx, ky], delta2_prime) + phi)) +
                    np.exp(1j * (np.dot([kx, ky], delta3_prime) + phi)))

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([[m + 2 * t2 * np.cos(phi), t1_term],
                            [np.conj(t1_term), -m + 2 * t2 * np.cos(phi)]])

    return hamiltonian


# Background: The Chern number is a topological invariant that characterizes the global properties of a band structure in a periodic system. 
# In the context of the Haldane model, it is used to determine the topological phase of the system. The Chern number is calculated by integrating 
# the Berry curvature over the entire Brillouin zone. For a discretized Brillouin zone, this involves summing the Berry curvature contributions 
# from each small plaquette. The Berry curvature can be computed using the eigenstates of the Hamiltonian at each point in the Brillouin zone. 
# The Chern number is expected to be an integer, indicating the number of edge states in the system.




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
        delta1 = np.array([1, 0]) * a
        delta2 = np.array([-0.5, sqrt(3)/2]) * a
        delta3 = np.array([-0.5, -sqrt(3)/2]) * a

        t1_term = t1 * (np.exp(1j * np.dot([kx, ky], delta1)) +
                        np.exp(1j * np.dot([kx, ky], delta2)) +
                        np.exp(1j * np.dot([kx, ky], delta3)))

        delta1_prime = delta1 + delta2
        delta2_prime = delta2 + delta3
        delta3_prime = delta3 + delta1

        t2_term = t2 * (np.exp(1j * (np.dot([kx, ky], delta1_prime) + phi)) +
                        np.exp(1j * (np.dot([kx, ky], delta2_prime) + phi)) +
                        np.exp(1j * (np.dot([kx, ky], delta3_prime) + phi)))

        hamiltonian = np.array([[m + 2 * t2 * np.cos(phi), t1_term],
                                [np.conj(t1_term), -m + 2 * t2 * np.cos(phi)]])
        return hamiltonian

    def berry_curvature(kx, ky, a, t1, t2, phi, m):
        hamiltonian = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        u = eigenvectors[:, 0]  # Ground state eigenvector
        u_kx = calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m)
        u_ky = calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m)
        u_kx_ky = calc_hamiltonian(kx + delta, ky + delta, a, t1, t2, phi, m)

        u_kx = np.linalg.eigh(u_kx)[1][:, 0]
        u_ky = np.linalg.eigh(u_ky)[1][:, 0]
        u_kx_ky = np.linalg.eigh(u_kx_ky)[1][:, 0]

        F12 = cmath.log(np.vdot(u, u_kx) * np.vdot(u_kx, u_kx_ky) * np.vdot(u_kx_ky, u_ky) * np.vdot(u_ky, u))
        return F12.imag

    kx_vals = np.arange(-pi/a, pi/a, delta)
    ky_vals = np.arange(-pi/a, pi/a, delta)
    chern_number = 0.0

    for kx in kx_vals:
        for ky in ky_vals:
            chern_number += berry_curvature(kx, ky, a, t1, t2, phi, m)

    chern_number /= (2 * pi)
    return chern_number



# Background: The Chern number is a topological invariant that characterizes the global properties of a band structure in a periodic system. 
# In the context of the Haldane model, it is used to determine the topological phase of the system. By sweeping the parameters such as the 
# on-site energy to next-nearest-neighbor coupling ratio (m/t2) and the phase (phi), we can explore how the Chern number changes across 
# different parameter regimes. This involves calculating the Chern number for each combination of these parameters over a discretized 
# Brillouin zone. The results can be visualized as a 2D array where each entry corresponds to a specific pair of (m/t2, phi) values.




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

        t1_term = t1 * (np.exp(1j * np.dot([kx, ky], delta1)) +
                        np.exp(1j * np.dot([kx, ky], delta2)) +
                        np.exp(1j * np.dot([kx, ky], delta3)))

        delta1_prime = delta1 + delta2
        delta2_prime = delta2 + delta3
        delta3_prime = delta3 + delta1

        t2_term = t2 * (np.exp(1j * (np.dot([kx, ky], delta1_prime) + phi)) +
                        np.exp(1j * (np.dot([kx, ky], delta2_prime) + phi)) +
                        np.exp(1j * (np.dot([kx, ky], delta3_prime) + phi)))

        hamiltonian = np.array([[m + 2 * t2 * np.cos(phi), t1_term],
                                [np.conj(t1_term), -m + 2 * t2 * np.cos(phi)]])
        return hamiltonian

    def berry_curvature(kx, ky, a, t1, t2, phi, m):
        hamiltonian = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        u = eigenvectors[:, 0]  # Ground state eigenvector
        u_kx = calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m)
        u_ky = calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m)
        u_kx_ky = calc_hamiltonian(kx + delta, ky + delta, a, t1, t2, phi, m)

        u_kx = np.linalg.eigh(u_kx)[1][:, 0]
        u_ky = np.linalg.eigh(u_ky)[1][:, 0]
        u_kx_ky = np.linalg.eigh(u_kx_ky)[1][:, 0]

        F12 = cmath.log(np.vdot(u, u_kx) * np.vdot(u_kx, u_kx_ky) * np.vdot(u_kx_ky, u_ky) * np.vdot(u_ky, u))
        return F12.imag

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
                    chern_number += berry_curvature(kx, ky, a, t1, t2, phi, m)

            chern_number /= (2 * pi)
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
