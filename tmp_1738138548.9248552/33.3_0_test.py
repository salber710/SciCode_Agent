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





def compute_chern_number(delta, a, t1, t2, phi, m):
    def haldane_hamiltonian(kx, ky):
        # Reciprocal lattice vectors
        b1 = np.array([2 * pi / (a * np.sqrt(3)), 2 * pi / (3 * a)])
        b2 = np.array([2 * pi / (a * np.sqrt(3)), -2 * pi / (3 * a)])
        
        # Nearest neighbor vectors in k-space
        nn_k = [np.dot([kx, ky], b1), np.dot([kx, ky], b1 - b2), np.dot([kx, ky], -b2)]
        
        # Next-nearest neighbor vectors in k-space
        nnn_k = [np.dot([kx, ky], b1*2), np.dot([kx, ky], -b1*2), np.dot([kx, ky], (b1-b2)*2), 
                 np.dot([kx, ky], -(b1-b2)*2), np.dot([kx, ky], b2*2), np.dot([kx, ky], -b2*2)]
        
        # Hamiltonian components
        re_h = m + 2 * t2 * sum(cos(k + phi) for k in nnn_k)
        im_h = t1 * sum(exp(1j * k) for k in nn_k)
        
        # Hamiltonian matrix
        h = np.array([[re_h, im_h], [np.conj(im_h), -re_h]])
        return h

    # Discretize the Brillouin zone
    kx_vals = np.linspace(-pi/a, pi/a, int(2*pi/(a*delta)))
    ky_vals = np.linspace(-pi/a, pi/a, int(2*pi/(a*delta)))
    
    chern_number = 0
    for i in range(len(kx_vals) - 1):
        for j in range(len(ky_vals) - 1):
            # Calculate Hamiltonians at the corners of a plaquette
            H00 = haldane_hamiltonian(kx_vals[i], ky_vals[j])
            H10 = haldane_hamiltonian(kx_vals[i+1], ky_vals[j])
            H01 = haldane_hamiltonian(kx_vals[i], ky_vals[j+1])
            H11 = haldane_hamiltonian(kx_vals[i+1], ky_vals[j+1])
            
            # Eigenstates at the corners
            _, psi00 = eigh(H00)
            _, psi10 = eigh(H10)
            _, psi01 = eigh(H01)
            _, psi11 = eigh(H11)
            
            # Berry connection phases
            U1 = np.vdot(psi00[:, 0], psi10[:, 0])
            U2 = np.vdot(psi10[:, 0], psi11[:, 0])
            U3 = np.vdot(psi11[:, 0], psi01[:, 0])
            U4 = np.vdot(psi01[:, 0], psi00[:, 0])
            
            # Berry curvature
            F_ij = np.log(U1 * U2 * U3 * U4).imag
            
            # Sum over the plaquettes
            chern_number += F_ij
    
    # Normalize the Chern number
    chern_number /= (2 * pi)
    
    return chern_number



# Background: The Chern number is a topological invariant that characterizes the global properties of a band structure in a periodic system. 
# In the context of the Haldane model, it is used to identify different topological phases. By varying parameters such as the on-site energy 
# to next-nearest-neighbor coupling ratio (m/t2) and the phase (phi), we can explore how the topological nature of the system changes. 
# The Chern number is computed by integrating the Berry curvature over the Brillouin zone, which can be discretized into a grid. 
# This function will compute a 2D array of Chern numbers by sweeping these parameters over specified ranges.





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

    # Define the range of m/t2 and phi values
    m_values = np.linspace(-6, 6, N)
    phi_values = np.linspace(-pi, pi, N)

    # Initialize the results matrix
    results = np.zeros((N, N))

    # Function to compute the Chern number for given m and phi
    def compute_chern_number(m, phi):
        def haldane_hamiltonian(kx, ky):
            # Reciprocal lattice vectors
            b1 = np.array([2 * pi / (a * sqrt(3)), 2 * pi / (3 * a)])
            b2 = np.array([2 * pi / (a * sqrt(3)), -2 * pi / (3 * a)])
            
            # Nearest neighbor vectors in k-space
            nn_k = [np.dot([kx, ky], b1), np.dot([kx, ky], b1 - b2), np.dot([kx, ky], -b2)]
            
            # Next-nearest neighbor vectors in k-space
            nnn_k = [np.dot([kx, ky], b1*2), np.dot([kx, ky], -b1*2), np.dot([kx, ky], (b1-b2)*2), 
                     np.dot([kx, ky], -(b1-b2)*2), np.dot([kx, ky], b2*2), np.dot([kx, ky], -b2*2)]
            
            # Hamiltonian components
            re_h = m + 2 * t2 * sum(cos(k + phi) for k in nnn_k)
            im_h = t1 * sum(cmath.exp(1j * k) for k in nn_k)
            
            # Hamiltonian matrix
            h = np.array([[re_h, im_h], [np.conj(im_h), -re_h]])
            return h

        # Discretize the Brillouin zone
        kx_vals = np.linspace(-pi/a, pi/a, int(2*pi/(a*delta)))
        ky_vals = np.linspace(-pi/a, pi/a, int(2*pi/(a*delta)))
        
        chern_number = 0
        for i in range(len(kx_vals) - 1):
            for j in range(len(ky_vals) - 1):
                # Calculate Hamiltonians at the corners of a plaquette
                H00 = haldane_hamiltonian(kx_vals[i], ky_vals[j])
                H10 = haldane_hamiltonian(kx_vals[i+1], ky_vals[j])
                H01 = haldane_hamiltonian(kx_vals[i], ky_vals[j+1])
                H11 = haldane_hamiltonian(kx_vals[i+1], ky_vals[j+1])
                
                # Eigenstates at the corners
                _, psi00 = eigh(H00)
                _, psi10 = eigh(H10)
                _, psi01 = eigh(H01)
                _, psi11 = eigh(H11)
                
                # Berry connection phases
                U1 = np.vdot(psi00[:, 0], psi10[:, 0])
                U2 = np.vdot(psi10[:, 0], psi11[:, 0])
                U3 = np.vdot(psi11[:, 0], psi01[:, 0])
                U4 = np.vdot(psi01[:, 0], psi00[:, 0])
                
                # Berry curvature
                F_ij = np.log(U1 * U2 * U3 * U4).imag
                
                # Sum over the plaquettes
                chern_number += F_ij
        
        # Normalize the Chern number
        chern_number /= (2 * pi)
        
        return chern_number

    # Sweep over m/t2 and phi values
    for i, m_ratio in enumerate(m_values):
        for j, phi in enumerate(phi_values):
            m = m_ratio * t2
            results[i, j] = compute_chern_number(m, phi)

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