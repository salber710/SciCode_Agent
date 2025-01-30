import numpy as np

# Background: In a semi-infinite system of layered electron gas (LEG), the Coulomb interaction between two electrons
# is influenced by the dielectric constant of the material and the spatial separation between the layers. The interaction
# potential can be expressed in terms of a form factor, which is a function of the in-plane momentum transfer `q` and
# the positions of the electrons in different layers, denoted by `l1` and `l2`. The form factor `f(q;z,z')` is derived
# by Fourier transforming the Coulomb interaction with respect to the in-plane coordinates. The dielectric constant
# `bg_eps` modifies the interaction strength, and the layer spacing `d` determines the vertical separation between
# the layers. The form factor is crucial for understanding the screening effects and electronic properties of the LEG.


def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1,l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    # Validate input parameters
    if l1 < 0 or l2 < 0:
        raise ValueError("Layer numbers l1 and l2 must be non-negative integers.")
    if not isinstance(l1, int) or not isinstance(l2, int):
        raise TypeError("Layer numbers l1 and l2 must be integers.")
    if d < 0:
        raise ValueError("Layer spacing d must be non-negative.")
    if bg_eps <= 0:
        raise ValueError("Dielectric constant bg_eps must be positive.")

    # Calculate the vertical separation between the layers
    z1 = l1 * d
    z2 = l2 * d
    delta_z = np.abs(z1 - z2)
    
    # Calculate the form factor using the exponential decay due to the separation
    # and the dielectric constant. The form factor is typically an exponential function
    # of the separation scaled by the in-plane momentum and the dielectric constant.
    form_factor = np.exp(-np.abs(q) * delta_z / bg_eps)
    
    return form_factor


# Background: In a two-dimensional electron gas (2DEG), the density-density correlation function, also known as the 
# Lindhard function, describes how the electron density responds to external perturbations. At zero temperature (T=0),
# this function can be derived using the properties of the Fermi surface. The correlation function D^0(q, ω + iγ) is 
# computed by integrating over the occupied states in the 2DEG. The real part of the energy is ω, and the imaginary 
# part is γ, which accounts for damping. The effective electron density n_eff, Fermi energy e_F, Fermi momentum k_F, 
# and Fermi velocity v_F are key parameters that define the electronic properties of the 2DEG. The integral involves 
# the difference between the energy of the perturbed state and the Fermi energy, and it is evaluated over the 
# momentum space up to the Fermi momentum.


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
    # Check for non-physical parameters
    if k_F < 0:
        raise ValueError("Fermi momentum k_F must be non-negative")
    if n_eff < 0:
        raise ValueError("Electron density n_eff must be non-negative")

    # Calculate the complex frequency
    omega_complex = omega + 1j * gamma
    
    # Define the integration limits
    k_min = 0
    k_max = k_F
    
    # Define the integrand for the density-density correlation function
    def integrand(k):
        # Energy difference between the perturbed state and the Fermi energy
        energy_diff = omega_complex - (v_F * (k + q) - e_F)
        # Avoid division by zero by adding a small imaginary part to the denominator
        return k / (energy_diff + 1e-10j)
    
    # Perform the integration over the momentum space
    integral_result = np.trapz(integrand(np.linspace(k_min, k_max, 1000)), np.linspace(k_min, k_max, 1000))
    
    # Normalize the result by the density of states at the Fermi level
    D0 = n_eff / (2 * np.pi) * integral_result
    
    return D0


# Background: In the Random Phase Approximation (RPA), the density-density correlation function for a layered electron gas (LEG)
# can be computed using matrix notation. The RPA accounts for the screening effects due to the Coulomb interaction between
# electrons in different layers. The Dyson equation in matrix form is used to compute the full correlation function D(l, l')
# from the non-interacting correlation function D0 and the Coulomb interaction. The matrix form of the Dyson equation is:
# D = D0 + D0 * V * D, where V is the Coulomb interaction matrix. The vacuum dielectric constant ε0 is used to scale the
# interaction strength. The matrix dimension N corresponds to the number of layers considered in the calculation.


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
    # Vacuum dielectric constant in the given units
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # Validate inputs
    if d < 0:
        raise ValueError("Layer spacing 'd' must be non-negative.")
    if bg_eps <= 0:
        raise ValueError("Background dielectric constant 'bg_eps' must be positive.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("Matrix dimension 'N' must be a positive integer.")
    if D0.shape != (N, N):
        raise ValueError("Dimension of D0 must match the matrix dimension N.")

    # Initialize the Coulomb interaction matrix V
    V = np.zeros((N, N), dtype=complex)

    # Fill the Coulomb interaction matrix V using the form factor
    for l1 in range(N):
        for l2 in range(N):
            # Calculate the form factor for the interaction between layers l1 and l2
            form_factor = np.exp(-np.abs(q) * np.abs(l1 - l2) * d / bg_eps)
            # Coulomb interaction scaled by the vacuum dielectric constant
            V[l1, l2] = (1 / (epsilon_0 * bg_eps)) * form_factor

    # Initialize the full correlation function matrix D
    D = np.zeros((N, N), dtype=complex)

    # Calculate the full correlation function using the Dyson equation in matrix form
    # D = D0 + D0 * V * D
    # This can be solved as a matrix equation: D = (I - D0 * V)^-1 * D0
    I = np.eye(N, dtype=complex)  # Identity matrix
    D = np.linalg.inv(I - np.dot(D0, V)).dot(D0)

    return D


# Background: In a semi-infinite layered electron gas (LEG), the density-density correlation function D(l, l') 
# describes the response of the electron density in one layer to a perturbation in another layer. Within the 
# Random Phase Approximation (RPA), this function can be derived using the Dyson equation, which incorporates 
# the effects of the Coulomb interaction between layers. The interaction is modified by the dielectric constant 
# of the material and the vacuum dielectric constant. The explicit form of D(l, l') can be obtained by solving 
# the Dyson equation analytically for a semi-infinite system, where the layers extend infinitely in one direction.

def D_l_analy(l1, l2, q, d, D0, bg_eps):
    '''Calculate the explicit form of density-density correlation function D(l1,l2) of semi-infinite LEG
    Input
    l1,l2: layer number where z = l*d, integer
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    bg_eps: LEG dielectric constant, float
    Output
    D_l: density-density correlation function, complex number in the unit of per square angstrom per meV
    '''


    # Vacuum dielectric constant in the given units
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # Validate inputs
    if l1 < 0 or l2 < 0:
        raise ValueError("Layer numbers l1 and l2 must be non-negative integers.")
    if d <= 0:  # Changed from < 0 to <= 0 to handle zero spacing case
        raise ValueError("Layer spacing 'd' must be positive.")
    if bg_eps <= 0:
        raise ValueError("Background dielectric constant 'bg_eps' must be positive.")

    # Ensure D0 has sufficient dimensions to access D0[l1, l2]
    if l1 >= D0.shape[0] or l2 >= D0.shape[1]:
        raise IndexError("Layer indices l1 or l2 are out of bounds for the given D0 matrix.")

    # Calculate the form factor for the interaction between layers l1 and l2
    form_factor = np.exp(-np.abs(q) * np.abs(l1 - l2) * d / bg_eps)

    # Coulomb interaction scaled by the vacuum dielectric constant
    V_l1_l2 = (1 / (epsilon_0 * bg_eps)) * form_factor

    # Calculate the explicit form of the density-density correlation function using the Dyson equation
    # D(l1, l2) = D0(l1, l2) + sum over l' [D0(l1, l') * V(l', l2) * D(l', l2)]
    # For a semi-infinite system, this simplifies to:
    D_l = D0[l1, l2] / (1 - D0[l1, l2] * V_l1_l2)

    return D_l



# Background: In a semi-infinite layered electron gas (LEG), surface plasmons are collective oscillations of the electron
# density that occur at the interface between the LEG and vacuum. The surface plasmon frequency, ω_s(q), depends on the
# in-plane momentum q and the electronic properties of the LEG, such as the effective electron density n_eff, Fermi energy
# e_F, Fermi momentum k_F, and Fermi velocity v_F. The Random Phase Approximation (RPA) is used to account for the screening
# effects due to the Coulomb interaction. The surface plasmon frequency can be determined by finding the poles of the
# density-density correlation function D(l, l') in the complex frequency plane. The vacuum dielectric constant ε0 is used
# to scale the interaction strength.


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

    # Vacuum dielectric constant in the given units
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # Calculate the 2D density-density correlation function D0(q, omega + i*gamma)
    D0 = D_2DEG(q, 0, gamma, n_eff, e_F, k_F, v_F)

    # The surface plasmon frequency is found by solving the condition for the poles of the RPA dielectric function
    # 1 - V(q) * D0(q, omega_s + i*gamma) = 0
    # where V(q) is the effective Coulomb interaction in 2D

    # Effective 2D Coulomb interaction
    V_q = 2 * np.pi / (q * epsilon_0 * bg_eps)

    # Solve for omega_s by finding the root of the real part of the dielectric function
    def dielectric_function(omega):
        D0_omega = D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F)
        return 1 - V_q * D0_omega

    # Use a numerical method to find the root of the dielectric function
    # Here, we use a simple bisection method or similar root-finding technique
    omega_min = 0
    omega_max = 2 * e_F  # A reasonable upper bound for the search
    omega_s = (omega_min + omega_max) / 2

    # Simple iterative method to find the root
    for _ in range(100):  # Limit the number of iterations
        if np.real(dielectric_function(omega_s)) > 0:
            omega_min = omega_s
        else:
            omega_max = omega_s
        omega_s = (omega_min + omega_max) / 2

    return omega_s

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('69.5', 3)
target = targets[0]

n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
m_eff = 0.07 ###unit: m_e (electron mass)
e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
d = 890   ### unit: A
bg_eps = 13.1
q = 0.05*k_F
gamma = 0
assert np.allclose(omega_s_cal(q,gamma,n_eff,e_F,k_F,v_F,d,bg_eps), target)
target = targets[1]

n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
m_eff = 0.07 ###unit: m_e (electron mass)
e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
d = 890   ### unit: A
bg_eps = 13.1
q = 0.1*k_F
gamma = 0
assert np.allclose(omega_s_cal(q,gamma,n_eff,e_F,k_F,v_F,d,bg_eps), target)
target = targets[2]

n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
m_eff = 0.07 ###unit: m_e (electron mass)
e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
d = 890   ### unit: A
bg_eps = 13.1
q = 0.15*k_F
gamma = 0
assert np.allclose(omega_s_cal(q,gamma,n_eff,e_F,k_F,v_F,d,bg_eps), target)
