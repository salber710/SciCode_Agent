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

    # Define the Coulomb interaction term for 2D
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)

    # Create the interaction matrix V with a new approach
    V = np.zeros((N, N), dtype=complex)

    # Using a sinusoidal modulation for the form factor
    for l1 in range(N):
        for l2 in range(N):
            form_factor = np.sin(q * d * (l1 + l2 + 1)) / (1 + (l1 - l2)**2)
            V[l1, l2] = V_q * form_factor

    # Calculate the interacting density-density correlation function D
    # Use a different approach to compute the inverse
    I = np.eye(N, dtype=complex)  # Identity matrix
    D = np.linalg.solve(I - np.dot(D0, V), D0)

    return D


try:
    targets = process_hdf5_to_tuple('69.3', 3)
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