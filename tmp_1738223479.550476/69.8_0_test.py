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


def omega_s_cal(q, gamma, n_eff, e_F, k_F, v_F, d, bg_eps):
    '''Calculate the surface plasmon of a semi-infinite LEG
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    Output
    omega_s: surface plasmon frequency, float in the unit of meV
    '''


    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # New approach: use a harmonic oscillator model to represent collective electron oscillations
    # This model assumes a simple harmonic potential influencing the electron density at the surface

    # Calculate the effective spring constant of the harmonic oscillator model
    k_eff = (2 * np.pi * n_eff * e_F) / (bg_eps * d**2)

    # Assume a mass term proportional to the Fermi energy and density
    mass_term = e_F * n_eff * epsilon_0 / (bg_eps * k_F)

    # Calculate the angular frequency of the harmonic oscillator
    omega_0 = np.sqrt(k_eff / mass_term)

    # Adjust the frequency for damping using the imaginary part gamma
    damping_adjustment = np.sqrt(1 + (gamma / omega_0)**2)

    # Final surface plasmon frequency
    omega_s = omega_0 / damping_adjustment

    return omega_s


def I_Raman(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd):
    '''Calculate the Raman intensity
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    bg_eps: LEG dielectric constant, float
    delta_E: penetration depth, float in the unit of angstrom
    kd: wave number times layer spacing, float dimensionless
    Output
    I_omega: Raman intensity, float
    '''


    # Calculate the complex frequency
    omega_complex = omega + 1j * gamma

    # Define a unique density-density correlation function with a Bessel function factor
    def D_bessel(q, omega_complex, n_eff, e_F, k_F, v_F):
        bessel_factor = np.i0(q / k_F)  # Modified Bessel function of the first kind
        return (n_eff * bessel_factor / (omega_complex - q * v_F)) / (1 + (q * bessel_factor / (bg_eps * e_F)) * (n_eff / (omega_complex - q * v_F)))

    # Calculate the Bessel-modified single layer correlation function
    D0 = D_bessel(q, omega_complex, n_eff, e_F, k_F, v_F)

    # Calculate the number of layers contributing based on penetration depth
    N_layers = int(delta_E / d)

    # Initialize Raman intensity
    I_omega = 0.0

    # Loop through pairs of layers and compute contributions
    for l1 in range(N_layers):
        for l2 in range(N_layers):
            # Calculate correlation between layers l1 and l2 with an exponential decay
            D_llp = D0 * np.exp(-kd * np.sqrt(np.abs(l1 - l2)))

            # Use a Heaviside step function as a weighting function to introduce sharp transitions
            weight = np.heaviside(kd * (l1 - l2) * d / delta_E, 1)

            # Sum the imaginary part of the correlation function, weighted by the factor
            I_omega += weight * np.imag(D_llp)

    return I_omega


def I_Raman_eval(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd):


    # Constants
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1
    alpha = 1.0  # Assumed constant

    # Complex frequency representation
    omega_complex = omega + 1j * gamma

    # Compute density-density correlation function D0
    D0 = n_eff / (np.pi * v_F**2 * (omega_complex - q * v_F))

    # Coulomb interaction term V_q
    V_q = (2 * np.pi * epsilon_0) / (q * bg_eps)

    # Precompute common terms
    sinh_qd = np.sinh(q * d)
    cosh_qd = np.cosh(q * d)
    exp_d_delta = np.exp(d / delta_E)
    exp_2d_delta = exp_d_delta**2
    cos_2kd = np.cos(2 * kd * d)

    # Calculate b
    b = cosh_qd - D0 * V_q * sinh_qd

    # Calculate u using a different method to ensure correct branch
    u = b + np.lib.scimath.sqrt(b**2 - 1)

    # G and H calculations using alternative approach
    G = 0.5 * (np.log1p((b**2 - 1)**0.5) - 1 / sinh_qd) / sinh_qd
    H = 0.5 * (1 / u * np.log1p((b**2 - 1)**0.5) - exp_d_delta / sinh_qd) / sinh_qd

    # A, B, C term calculations
    A = G * sinh_qd**2 + 1 + 0.5 * alpha * np.exp(2 * q * d)
    B = H * sinh_qd**2 + cosh_qd + 0.5 * alpha * np.exp(q * d)
    C = G * sinh_qd**2 + 1 + 0.5 * alpha

    # Calculate Q with altered form
    Q = 0.5 * (1 - (b**2 - 1)**-0.5 * (1 - b * cosh_qd) / sinh_qd) \
        - 0.5 * alpha * exp_d_delta * (b**2 - 1)**-0.5 * (cosh_qd - b) / sinh_qd

    # Calculate E expression
    E = u**2 * exp_2d_delta + 1 - 2 * u * exp_d_delta * cos_2kd

    # Compute the Raman intensity I(omega)
    prefactor = (1 - np.exp(-2 * d / delta_E))**-1
    term1 = 1 + (D0 * V_q * sinh_qd * (u**2 * exp_2d_delta - 1)) / (np.lib.scimath.sqrt(b**2 - 1) * E)
    term2 = (D0 * V_q * exp_2d_delta * (u**2 * A - 2 * u * B + C)) / (2 * Q * np.lib.scimath.sqrt(b**2 - 1) * E)

    I_omega = -np.imag(D0 * prefactor * (term1 + term2))

    return I_omega



# Background: In the context of a semi-infinite layered electron gas (LEG), the density-density correlation
# function within the Random Phase Approximation (RPA) framework describes the response of the electron system 
# to external perturbations. This function, D(l,l'), accounts for interactions between electron layers.
# Numerically, we calculate this correlation function using matrix algebra, where the Coulomb interaction 
# acts as a self-energy term in the Dyson equation. The Raman intensity, which measures inelastic scattering
# of light by the system, is then derived from the imaginary part of the correlation function, indicating 
# energy losses due to excitations in the system.

def I_Raman_num(q, d, omega, gamma, n_eff, e_F, k_F, v_F, bg_eps, delta_E, kd, N):
    '''Calculate the Raman intensity
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    bg_eps: LEG dielectric constant, float
    delta_E: penetration depth, float in the unit of angstrom
    kd: wave number times layer spacing, float dimensionless
    N: matrix dimension, integer
    Output
    I_omega_num: Raman intensity, float
    '''


    # Convert omega and gamma to a complex frequency
    omega_complex = omega + 1j * gamma

    # Calculate the prefactor for the density-density correlation function
    prefactor = n_eff / (np.pi * v_F**2)

    # Calculate the dimensionless variable lambda_q
    lambda_q = q / k_F

    # Lindhard function for 2D electron gas
    def lindhard_2d(lambda_q, sigma):
        if lambda_q < 1:
            return 1 - lambda_q**2 - sigma**2
        else:
            return 1 - lambda_q**2 - sigma**2

    # Compute D0, the non-interacting density-density correlation function
    sigma = omega_complex / (v_F * q)
    F = lindhard_2d(lambda_q, sigma)
    D0 = prefactor * F

    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # Compute the 2D Coulomb interaction term V_q
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)

    # Construct the interaction matrix V
    V = np.zeros((N, N), dtype=complex)
    for l1 in range(N):
        for l2 in range(N):
            # Using a basic form factor here
            form_factor = 1 / (1 + (l1 - l2)**2)
            V[l1, l2] = V_q * form_factor

    # Calculate the interacting density-density correlation function D using matrix inversion
    I = np.eye(N, dtype=complex)  # Identity matrix
    A = I - np.dot(V, D0)

    # Solve for D using matrix inversion
    D = np.linalg.inv(A) @ D0

    # Calculate the Raman intensity
    I_omega_num = 0.0
    for l1 in range(N):
        for l2 in range(N):
            D_llp = D[l1, l2]
            weight = np.heaviside(kd * (l1 - l2) * d / delta_E, 1)
            I_omega_num += weight * np.imag(D_llp)

    return I_omega_num


try:
    targets = process_hdf5_to_tuple('69.8', 4)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    gamma = 0.1
    q = 0.04669*k_F
    omega = 0.6
    delta_E = 6000
    kd = 4.94/2
    N = 1000
    assert np.allclose(I_Raman_num(q,d,omega,gamma,n_eff,e_F,k_F,v_F,bg_eps,delta_E,kd,N), target)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    gamma = 0.1
    q = 0.04669*k_F
    omega = 7
    delta_E = 6000
    kd = 4.94/2
    N = 1000
    assert np.allclose(I_Raman_num(q,d,omega,gamma,n_eff,e_F,k_F,v_F,bg_eps,delta_E,kd,N), target)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    gamma = 0.1
    q = 0.04669*k_F
    omega = 12.275
    delta_E = 6000
    kd = 4.94/2
    N = 1000
    assert np.allclose(I_Raman_num(q,d,omega,gamma,n_eff,e_F,k_F,v_F,bg_eps,delta_E,kd,N), target)

    target = targets[3]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    gamma = 0.1
    q = 0.04669*k_F
    omega = 12.275
    delta_E = 6000
    kd = 4.94/2
    N = 1000
    NUM = I_Raman_num(q,d,omega,gamma,n_eff,e_F,k_F,v_F,bg_eps,delta_E,kd,N)
    ANA = I_Raman(q,d,omega,gamma,n_eff,e_F,k_F,v_F,bg_eps,delta_E,kd)
    tol = 1e-6
    assert (np.abs(NUM-ANA)< tol*abs(ANA)) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e