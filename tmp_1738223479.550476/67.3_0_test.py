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



# Background: In the RPA (Random Phase Approximation), the density-density correlation function for a layered electron gas (LEG)
# can be expressed in matrix form, incorporating the Coulomb interaction between layers. The Dyson equation describes how the 
# correlation function D(l, l') relates to the non-interacting correlation function D0 and the interaction potential. 
# The matrix form allows capturing interactions between multiple layers efficiently. The Dyson equation in matrix form is:
# D = D0 + D0 * V * D, where V is the matrix of Coulomb interactions between layers. This can be solved using matrix algebra.
# The inverse dielectric function matrix is given by ε = 1 - V * D0, and the density-density correlation function is D = ε^{-1} * D0.


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
    
    # Precompute V_q, the 2D Fourier transform of the Coulomb interaction
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)
    
    # Initialize the Coulomb interaction matrix V
    V = np.zeros((N, N), dtype=complex)
    
    # Fill the Coulomb interaction matrix with form factors
    for l1 in range(N):
        for l2 in range(N):
            # Calculate the form factor for layers l1 and l2
            form_factor = np.arccosh(1 + q * abs(l1 - l2) * d) / (1 + np.sinh(q * d)) / bg_eps
            V[l1, l2] = V_q * form_factor
    
    # Calculate the inverse dielectric matrix ε = 1 - V * D0
    identity_matrix = np.eye(N, dtype=complex)
    epsilon_matrix = identity_matrix - np.dot(V, D0)
    
    # Invert the dielectric matrix to find D = ε^{-1} * D0
    epsilon_inv = np.linalg.inv(epsilon_matrix)
    D = np.dot(epsilon_inv, D0)
    
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