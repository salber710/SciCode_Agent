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

from scicode.parse.parse import process_hdf5_to_tuple
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
