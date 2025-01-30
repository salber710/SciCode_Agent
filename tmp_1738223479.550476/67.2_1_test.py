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

    # Convert q to a dimensionless parameter by dividing by k_F
    q_dimensionless = q / k_F

    # Calculate the Lindhard function for a 2DEG using an alternative approach
    # Here we directly compute the response function, avoiding piecewise conditions
    def response_function(q_dimensionless, complex_omega):
        u_plus = (q_dimensionless + complex_omega / e_F) / 2
        u_minus = (q_dimensionless - complex_omega / e_F) / 2

        if q_dimensionless < 2:
            real_part = n_eff / (2 * e_F) * (1 - u_plus * np.log(abs((1 + u_plus) / (1 - u_plus))))
            imag_part = -n_eff * np.pi / (2 * e_F) * np.heaviside(1 - u_plus**2, 0.5)
        else:
            real_part = n_eff / (2 * e_F) * (1 - u_minus * np.log(abs((u_minus + 1) / (u_minus - 1))))
            imag_part = 0

        return real_part + 1j * imag_part

    D0 = response_function(q_dimensionless, complex_omega)

    return D0


try:
    targets = process_hdf5_to_tuple('67.2', 3)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    q = 0.1*k_F
    omega = 0.1*e_F
    gamma = 0
    assert np.allclose(D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F), target)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    q = 1*k_F
    omega = 0.5*e_F
    gamma = 0
    assert np.allclose(D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F), target)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    q = 3*k_F
    omega = 1*e_F
    gamma = 0
    assert np.allclose(D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e