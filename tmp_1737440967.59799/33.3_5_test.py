import numpy as np
import cmath
from math import pi, sin, cos, sqrt

# Background: The Haldane model describes electrons on a hexagonal lattice with both nearest-neighbor (NN) and next-nearest-neighbor (NNN) hopping. 
# It incorporates a complex phase in the NNN hopping term, which can lead to topologically nontrivial band structures. The Hamiltonian for the Haldane model is a 2x2 matrix.
# The NN hopping introduces off-diagonal terms in the Hamiltonian, while the NNN hopping contributes both to diagonal terms (with a phase) and an on-site energy term.
# The wavevector components (kx, ky) are part of the reciprocal space representation, and they affect the phase of the hopping terms due to the Bloch's theorem.
# The lattice structure of the hexagonal lattice contributes to the specific form of the Hamiltonian matrix.




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

    # Define the lattice vectors for the hexagonal lattice
    a1 = a * np.array([1, 0])
    a2 = a * np.array([-0.5, sqrt(3)/2])
    a3 = a * np.array([-0.5, -sqrt(3)/2])

    # Calculate the phase factors for nearest-neighbor hopping
    delta1 = np.array([1, 0])
    delta2 = np.array([-0.5, sqrt(3)/2])
    delta3 = np.array([-0.5, -sqrt(3)/2])

    # Calculate the nearest-neighbor contributions
    f_k = t1 * (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
                np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
                np.exp(1j * (kx * delta3[0] + ky * delta3[1])))

    # Calculate next-nearest-neighbor contributions with phase
    g_k = 2 * t2 * (cos(kx * a1[0] + ky * a1[1] - phi) +
                    cos(kx * a2[0] + ky * a2[1] - phi) +
                    cos(kx * a3[0] + ky * a3[1] - phi))

    # Construct the Haldane Hamiltonian matrix
    hamiltonian = np.array([[m + g_k, f_k],
                            [np.conjugate(f_k), -(m + g_k)]])
    
    return hamiltonian


# Background: The Chern number is a topological invariant that characterizes the topology of electronic band structures in condensed matter physics. 
# In the context of the Haldane model, it reflects the presence of topologically protected edge states. To calculate the Chern number, one must integrate 
# the Berry curvature over the Brillouin zone. For discrete systems, this involves summing the Berry curvature over a mesh grid in the momentum space. 
# The Berry curvature can be derived from the eigenvectors of the Hamiltonian, and the Chern number is obtained from the integral over the Brillouin zone, 
# which should be quantized to an integer value if the system exhibits a nontrivial topology.




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

    # Function to calculate the Hamiltonian for given kx, ky
    def calc_hamiltonian(kx, ky):
        a1 = a * np.array([1, 0])
        a2 = a * np.array([-0.5, sqrt(3)/2])
        a3 = a * np.array([-0.5, -sqrt(3)/2])

        delta1 = np.array([1, 0])
        delta2 = np.array([-0.5, sqrt(3)/2])
        delta3 = np.array([-0.5, -sqrt(3)/2])

        f_k = t1 * (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
                    np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
                    np.exp(1j * (kx * delta3[0] + ky * delta3[1])))

        g_k = 2 * t2 * (cos(kx * a1[0] + ky * a1[1] - phi) +
                        cos(kx * a2[0] + ky * a2[1] - phi) +
                        cos(kx * a3[0] + ky * a3[1] - phi))

        hamiltonian = np.array([[m + g_k, f_k],
                                [np.conjugate(f_k), -(m + g_k)]])
        return hamiltonian

    # Discretize the Brillouin zone
    kx_vals = np.arange(-pi/a, pi/a, delta)
    ky_vals = np.arange(-pi/a, pi/a, delta)
    
    chern_sum = 0

    # Compute the Berry curvature over the Brillouin zone
    for kx in kx_vals:
        for ky in ky_vals:
            hamiltonian = calc_hamiltonian(kx, ky)
            eigvals, eigvecs = np.linalg.eigh(hamiltonian)
            u = eigvecs[:, 0]  # Ground state eigenvector

            # Calculate Berry curvature
            d_ux = np.gradient(u, delta, axis=0)
            d_uy = np.gradient(u, delta, axis=1)
            berry_curvature = 2 * np.imag(np.vdot(d_ux, d_uy))

            chern_sum += berry_curvature * (delta**2)

    # The Chern number is the integral of the Berry curvature over the BZ
    chern_number = chern_sum / (2 * pi)
    
    return np.real(chern_number)



# Background: In the study of topological phases of matter, the Chern number is a crucial topological invariant that characterizes the band structure of a system.
# For the Haldane model on a hexagonal lattice, the Chern number can vary with different parameters such as the on-site energy to next-nearest-neighbor coupling ratio (m/t2) and the phase (phi).
# By sweeping these parameters across specified ranges, we can construct a 2D array of Chern numbers to explore how the topology of the system changes. This involves computing the Chern number for 
# each pair of (m/t2, phi) values using the previously defined method for calculating the Chern number over the discretized Brillouin zone.




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
    
    def compute_chern_number(delta, a, t1, t2, phi, m):
        # Function to calculate the Hamiltonian for given kx, ky
        def calc_hamiltonian(kx, ky):
            a1 = a * np.array([1, 0])
            a2 = a * np.array([-0.5, sqrt(3)/2])
            a3 = a * np.array([-0.5, -sqrt(3)/2])

            delta1 = np.array([1, 0])
            delta2 = np.array([-0.5, sqrt(3)/2])
            delta3 = np.array([-0.5, -sqrt(3)/2])

            f_k = t1 * (np.exp(1j * (kx * delta1[0] + ky * delta1[1])) +
                        np.exp(1j * (kx * delta2[0] + ky * delta2[1])) +
                        np.exp(1j * (kx * delta3[0] + ky * delta3[1])))

            g_k = 2 * t2 * (cos(kx * a1[0] + ky * a1[1] - phi) +
                            cos(kx * a2[0] + ky * a2[1] - phi) +
                            cos(kx * a3[0] + ky * a3[1] - phi))

            hamiltonian = np.array([[m + g_k, f_k],
                                    [np.conjugate(f_k), -(m + g_k)]])
            return hamiltonian

        # Discretize the Brillouin zone
        kx_vals = np.arange(-pi/a, pi/a, delta)
        ky_vals = np.arange(-pi/a, pi/a, delta)
        
        chern_sum = 0

        # Compute the Berry curvature over the Brillouin zone
        for kx in kx_vals:
            for ky in ky_vals:
                hamiltonian = calc_hamiltonian(kx, ky)
                eigvals, eigvecs = np.linalg.eigh(hamiltonian)
                u = eigvecs[:, 0]  # Ground state eigenvector

                # Calculate Berry curvature
                d_ux = np.gradient(u, delta, axis=0)
                d_uy = np.gradient(u, delta, axis=1)
                berry_curvature = 2 * np.imag(np.vdot(d_ux, d_uy))

                chern_sum += berry_curvature * (delta**2)

        # The Chern number is the integral of the Berry curvature over the BZ
        chern_number = chern_sum / (2 * pi)
        
        return np.real(chern_number)

    # Generate the m/t2 and phi values to sweep over
    m_values = np.linspace(-6, 6, N)
    phi_values = np.linspace(-pi, pi, N)

    # Initialize the results array
    results = np.zeros((N, N))

    # Sweep over the parameter space
    for i, m_t2 in enumerate(m_values):
        m = m_t2 * t2
        for j, phi in enumerate(phi_values):
            results[i, j] = compute_chern_number(delta, a, t1, t2, phi, m)

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
