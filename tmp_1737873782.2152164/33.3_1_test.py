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

    # Discretize the Brillouin zone into a grid
    num_k_points = int((2 * np.pi) / delta)
    chern_sum = 0.0

    # Iterate over the grid to calculate the Berry curvature
    for i in range(num_k_points):
        for j in range(num_k_points):
            # Calculate wavevectors
            kx = i * delta
            ky = j * delta

            # Calculate Hamiltonian at k, k + delta in both directions
            H_k = calc_hamiltonian(kx, ky, a, t1, t2, phi, m)
            H_kx = calc_hamiltonian(kx + delta, ky, a, t1, t2, phi, m)
            H_ky = calc_hamiltonian(kx, ky + delta, a, t1, t2, phi, m)
            H_kx_ky = calc_hamiltonian(kx + delta, ky + delta, a, t1, t2, phi, m)

            # Calculate the Berry curvature using the Fukui-Hatsugai-Suzuki method
            U_k_kx = np.linalg.det(np.linalg.solve(H_k, H_kx))
            U_kx_ky = np.linalg.det(np.linalg.solve(H_kx, H_kx_ky))
            U_ky_k = np.linalg.det(np.linalg.solve(H_ky, H_k))
            U_k_ky = np.linalg.det(np.linalg.solve(H_k, H_ky))

            # Calculate the complex logarithm of the product of U matrices
            F_k = cmath.log(U_k_kx * U_kx_ky * U_ky_k * U_k_ky).imag

            # Sum the Berry curvature contributions over the Brillouin zone
            chern_sum += F_k

    # The Chern number is the sum of Berry curvature over the Brillouin zone divided by 2π
    chern_number = chern_sum / (2 * np.pi)

    return chern_number



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

    # Initialize arrays to store results and swept parameter values
    results = np.zeros((N, N))
    m_values = np.linspace(-6 * t2, 6 * t2, N)
    phi_values = np.linspace(-pi, pi, N)

    # Loop over the grid of (m, phi) parameters
    for i, m in enumerate(m_values):
        for j, phi in enumerate(phi_values):
            # Compute the Chern number for each parameter pair
            chern_number = compute_chern_number(delta, a, t1, t2, phi, m)
            results[i, j] = chern_number

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