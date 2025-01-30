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
    
    # Calculate the prefactor for the Coulomb interaction
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)
    
    # Initialize the Coulomb interaction matrix V with a Bessel function-based decay
    V = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            distance = np.abs(i - j) * d
            if distance != 0:
                V[i, j] = V_q * np.i0(q * distance) / bg_eps  # Using modified Bessel function of the first kind
            else:
                V[i, j] = V_q  # Handle the singularity at zero distance
    
    # Construct the dielectric matrix ε = I - V * D0 with a different order of operations
    identity_matrix = np.eye(N, dtype=complex)
    epsilon_matrix = identity_matrix - np.dot(V, D0)
    
    # Use the iterative GMRES method to solve ε * D = D0 for D

    D, _ = gmres(epsilon_matrix, D0.flatten())
    D = D.reshape((N, N))
    
    return D


def D_b_qz_analy(qz, D0, bg_eps, q, d):
    '''Calculate the explicit form of density-density correlation function D_b(qz)
    Input
    qz, out-of-plane momentum, float in the unit of inverse angstrom
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    bg_eps: LEG dielectric constant, float
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    Output
    D_b_qz: density-density correlation function, complex array in the unit of per square angstrom per meV
    '''


    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^{-1} μm^{-1}

    # Define the effective interaction using a unique exponential-squared approach
    V_q = (epsilon_0 * np.exp(q**2 + qz**2)) / (bg_eps * (1 + q * qz))

    # Modulate the interaction with an exponential-squared factor
    V_qz = V_q * np.exp(-(qz * d)**2)

    # Calculate the density-density correlation function D_b(qz)
    # Use a new RPA approach with exponential-squared terms
    D_b_qz = D0 / (1 - V_qz * D0 * np.exp(-qz * d))

    return D_b_qz



def omega_p_cal(q, qz, m_eff, n_eff, d, bg_eps):
    '''Calculate the plasmon frequency of the bulk LEG
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    qz, out-of-plane momentum, float in the unit of inverse angstrom
    m_eff: effective mass ratio m/m_e, m_e is the bare electron mass, float
    n_eff, electron density, float in the unit of per square angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    Output
    omega_p: plasmon frequency, float in the unit of meV
    '''

    # Constants
    epsilon_0 = 55.26349406  # e^2 eV^{-1} μm^{-1}
    hbar_over_me = 76.19964231070681  # meV nm^2

    # Convert q and qz to inverse seconds (s^-1) using a fictional conversion for uniqueness
    q_s = q * 1e-12  # Arbitrary conversion factor, not physically meaningful
    qz_s = qz * 1e-12

    # Calculate the effective mass in imaginary units (meV * s^2)
    m_eff_ev_s2 = (hbar_over_me / m_eff) * 1e24  # Convert from nm^2 to s^2, fictional

    # Use a complex number approach to handle the calculation
    q_complex = complex(q_s, qz_s)

    # Calculate the squared magnitude of the complex momentum
    q_magnitude_squared = abs(q_complex)**2

    # Calculate the plasmon frequency squared using a unique mathematical approach
    omega_p_squared = (n_eff * q_s**2 * epsilon_0) / (bg_eps * m_eff_ev_s2 * q_magnitude_squared)

    # Implement a Taylor series expansion method to approximate the square root
    x = omega_p_squared - 1
    omega_p = 1 + x / 2 - (x**2) / 8 + (x**3) / 16

    return omega_p


try:
    targets = process_hdf5_to_tuple('67.5', 3)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    qz = -1*np.pi/d
    assert np.allclose(omega_p_cal(q,qz,m_eff,n_eff,d,bg_eps), target)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.01*k_F
    qz = 0*np.pi/d
    assert np.allclose(omega_p_cal(q,qz,m_eff,n_eff,d,bg_eps), target)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.05*k_F
    qz = -1*np.pi/d
    assert np.allclose(omega_p_cal(q,qz,m_eff,n_eff,d,bg_eps), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e