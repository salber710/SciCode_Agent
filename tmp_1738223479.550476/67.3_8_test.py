from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1, l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''


    # Compute the z-coordinates for the two layers
    z1 = l1 * d
    z2 = l2 * d

    # Calculate the distance in the z-direction
    delta_z = abs(z2 - z1)

    # Implement a form factor using an inverse hyperbolic cosine function
    # This introduces a unique decay based on q and delta_z
    form_factor = np.arccosh(1 + q * delta_z) / (1 + np.sinh(q * d)) / bg_eps

    return form_factor




def D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F):
    '''Write down the exact form of density-density correlation function
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    Output
    D0: density-density correlation function, complex array in the unit of per square angstrom per meV
    '''

    # Compute the complex frequency ω + iγ
    complex_omega = omega + 1j * gamma

    # Convert q to a dimensionless parameter
    q_dimensionless = q / k_F

    # Distinct approach using residue theorem and poles analysis
    def residue_response(q_dimensionless, complex_omega):
        # Calculate poles in the complex plane
        def pole_integrand(k):
            # Energy dispersion and complex frequency shift
            epsilon_k = k**2 / (2 * e_F)
            delta_e = epsilon_k + complex_omega - e_F

            # Calculate residues at the poles
            if abs(delta_e) < 1e-10:
                # Avoid division by zero, return zero contribution for very small delta_e
                return 0
            else:
                return (k / (2 * np.pi)) * np.log(abs((delta_e + q_dimensionless * v_F) / (delta_e - q_dimensionless * v_F)))

        # Define integration limits over a relevant range of the Fermi surface
        k_min, k_max = 0, 2 * k_F

        # Perform integration over the poles
        real_integral, real_error = quad(lambda k: pole_integrand(k).real, k_min, k_max)
        imag_integral, imag_error = quad(lambda k: pole_integrand(k).imag, k_min, k_max)

        # Return the density-density correlation function
        return (n_eff / e_F) * (real_integral + 1j * imag_integral)

    # Calculate the density-density correlation function
    D0 = residue_response(q_dimensionless, complex_omega)

    return D0





def D_cal(D0, q, d, bg_eps, N):
    '''Calculate the matrix form of density-density correlation function D(l1,l2)
    Input
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    N: matrix dimension, integer
    Output
    D: NxN complex matrix, in the unit of per square angstrom per meV
    '''
    
    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^{-1} μm^{-1}
    
    # Calculate the prefactor for the Coulomb potential
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)
    
    # Initialize the Coulomb interaction matrix V using a cosine-based function
    V = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            distance = np.abs(i - j) * d
            if distance != 0:
                V[i, j] = V_q * np.cos(q * distance) / (1 + distance)
            else:
                V[i, j] = V_q  # Handle the singularity at zero distance
    
    # Construct the dielectric matrix ε = I - V * D0 using a different matrix product order
    identity_matrix = np.eye(N, dtype=complex)
    epsilon_matrix = identity_matrix - np.dot(D0, V)
    
    # Use an alternative method to solve ε * D = D0 for D, such as solving the triangular system
    # First perform LU decomposition on the inverted matrix for stability
    U, s, Vh = np.linalg.svd(epsilon_matrix)
    S_inv = np.diag(1/s)
    epsilon_inv = np.dot(Vh.T.conj(), np.dot(S_inv, U.T.conj()))
    
    # Solve the equation using solve_triangular for the upper triangular matrix
    D = solve_triangular(epsilon_inv, D0, lower=False)
    
    return D


try:
    targets = process_hdf5_to_tuple('67.3', 3)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 0.1*e_F
    gamma = 0
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    N = 100
    assert np.allclose(D_cal(D0,q,d,bg_eps,N), target)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 1*k_F
    omega = 0.5*e_F
    gamma = 0
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    N = 100
    assert np.allclose(D_cal(D0,q,d,bg_eps,N), target)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 1.5*k_F
    omega = 2*e_F
    gamma = 0
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    N = 100
    assert np.allclose(D_cal(D0,q,d,bg_eps,N), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e