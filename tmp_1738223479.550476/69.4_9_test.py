from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def f_V(q, d, bg_eps, l1, l2):
    '''Compute the form factor f(q;l1,l2) using a completely different approach
    Input
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1, l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    # Compute the z-coordinates for the given layers
    z1 = l1 * d
    z2 = l2 * d
    
    # Introduce a sigmoid function to smoothly interpolate the interaction
    sigmoid_factor = 1 / (1 + np.exp(-q * (z1 - z2) / d))
    
    # Use a Lambert W function-based term for a unique oscillatory behavior
    lambert_w_term = np.real(np.exp(np.log(q * d + 1) - np.log(bg_eps) + np.log(np.abs(z1 - z2) + 1)))
    
    # Implement a quadratic damping factor with a distinct scaling
    quadratic_damping = np.exp(-q**2 * (z1**2 + z2**2) / (bg_eps**2 * d**2))
    
    # Combine these terms to form the final form factor
    form_factor = sigmoid_factor * lambert_w_term * quadratic_damping

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

    # Convert omega and gamma to a complex frequency
    omega_complex = omega + 1j * gamma

    # Calculate the prefactor for the density-density correlation function
    prefactor = n_eff / (np.pi * v_F**2)

    # Calculate the dimensionless variables
    lambda_q = q / k_F
    sigma = omega_complex / (v_F * q)

    # Distinct approach using an exponential integral function
    def lindhard_exponential_integral(lambda_q, sigma):
        if lambda_q < 1:
            exp_integral = expi(-lambda_q * sigma)
            return 1 - lambda_q**2 - sigma**2 + lambda_q * exp_integral
        else:
            return 1 - lambda_q**2 - sigma**2

    # Calculate the Lindhard function
    F = lindhard_exponential_integral(lambda_q, sigma)

    # Calculate the density-density correlation function D0
    D0 = prefactor * F

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
    epsilon_0 = 55.26349406  # in units of e^2 eV^-1 μm^-1

    # Compute the 2D Coulomb interaction term
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)

    # Construct the interaction matrix V using a unique approach
    V = np.zeros((N, N), dtype=complex)

    # Use a Chebyshev polynomial based form factor for layer interactions

    for l1 in range(N):
        for l2 in range(N):
            form_factor = chebval(q * d * (l1 + l2), [0, 1, 0.5]) / (1 + (l1 - l2)**2)
            V[l1, l2] = V_q * form_factor

    # Calculate the interacting density-density correlation function D
    I = np.eye(N, dtype=complex)  # Identity matrix
    A = I - np.dot(V, D0)

    # Use Cholesky decomposition for stable inversion
    try:
        L = cholesky(A, lower=True)
        Y = solve_triangular(L, D0, lower=True)
        D = solve_triangular(L.T.conj(), Y, lower=False)
    except np.linalg.LinAlgError:
        # Fallback to standard inversion if Cholesky fails
        D = np.dot(np.linalg.inv(A), D0)

    return D



def D_l_analy(l1, l2, q, d, D0, bg_eps):
    '''Calculate the explicit form of density-density correlation function D(l1,l2) of semi-infinite LEG
    Input
    l1, l2: layer number where z = l*d, integer
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    D0: density-density correlation function, complex array in the unit of per square angstrom per meV
    bg_eps: LEG dielectric constant, float
    Output
    D_l: density-density correlation function, complex number in the unit of per square angstrom per meV
    '''



    # Vacuum dielectric constant
    epsilon_0 = 55.26349406

    # Compute the 2D Coulomb interaction using a hypergeometric scaling
    V_q = epsilon_0 * hyp2f1(0.5, 1.5, 2.0, -q**2 * d**2) / (np.sqrt(q * bg_eps))

    # Use Chebyshev polynomials of the second kind for the form factor calculation

    form_factor = chebval(q * d * (l1 - l2), [0, 1, 0.4, 0.3]) * chebval(q * d * (l1 + l2), [1, 0.2, 0.1])

    # Calculate the effective interaction using a combination of hyperbolic and trigonometric functions
    V_l1_l2 = V_q * np.sinh(form_factor) * np.cos(form_factor)

    # Compute the interacting density-density correlation function D_l
    # using a complex square root function for additional non-linearity
    D_l = D0 / (1 - np.sqrt(V_l1_l2 * D0 + 1j))

    return D_l


try:
    targets = process_hdf5_to_tuple('69.4', 3)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 2*e_F
    gamma = 0.3
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    l1 = 1
    l2 = 3
    assert np.allclose(D_l_analy(l1,l2,q,d,D0,bg_eps), target, atol=1e-13, rtol=1e-13)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 2*e_F
    gamma = 0.3
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    l1 = 10
    l2 = 12
    assert np.allclose(D_l_analy(l1,l2,q,d,D0,bg_eps), target, atol=1e-13, rtol=1e-13)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 2*e_F
    gamma = 0.3
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    l1 = 4
    l2 = 7
    assert np.allclose(D_l_analy(l1,l2,q,d,D0,bg_eps), target, atol=1e-13, rtol=1e-13)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e