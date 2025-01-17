import numpy as np

# Background: In a semi-infinite system of layered electron gas (LEG), the Coulomb interaction between two electrons
# is influenced by the dielectric constant of the material and the spatial separation between the layers. The interaction
# can be described in terms of a form factor, which is a function of the in-plane momentum transfer (q) and the positions
# of the electrons in the layers (z = l1*d and z' = l2*d). The form factor f(q;z,z') is derived by Fourier transforming
# the Coulomb potential with respect to the in-plane coordinates. The dielectric constant (bg_eps) modifies the effective
# interaction strength, and the layer spacing (d) determines the separation between the electron layers. The form factor
# is crucial for understanding the screening effects and the effective interaction in the layered system.


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
    # Calculate the distance between the layers in the z-direction
    z1 = l1 * d
    z2 = l2 * d
    delta_z = np.abs(z1 - z2)
    
    # Calculate the form factor using the exponential decay due to the layer separation
    form_factor = np.exp(-q * delta_z) / bg_eps
    
    return form_factor


# Background: In a two-dimensional electron gas (2DEG), the density-density correlation function, also known as the 
# Lindhard function, describes how the electron density responds to external perturbations. At zero temperature (T=0), 
# this function can be derived using the random phase approximation (RPA) and is crucial for understanding the 
# screening properties and collective excitations in the system. The correlation function D^0(q, ω + iγ) is a complex 
# function of the in-plane momentum transfer q, the energy ω, and a small imaginary part γ that ensures causality. 
# The parameters n_eff, e_F, k_F, and v_F represent the effective electron density, Fermi energy, Fermi momentum, 
# and Fermi velocity, respectively. The correlation function is computed by integrating over the occupied states 
# in the Fermi sea, taking into account the energy and momentum conservation laws.


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
    # Calculate the complex frequency
    omega_complex = omega + 1j * gamma
    
    # Calculate the Lindhard function for a 2DEG
    # The Lindhard function is given by:
    # D^0(q, omega) = (m / (2 * pi * hbar^2)) * [1 - (omega / (q * v_F)) * ln((omega + q * v_F) / (omega - q * v_F))]
    # where m is the effective mass, which can be related to the Fermi velocity and Fermi momentum.
    
    # Effective mass m can be derived from the Fermi velocity and Fermi momentum:
    # m = hbar * k_F / v_F
    m = k_F / v_F
    
    # Calculate the prefactor
    prefactor = m / (2 * np.pi)
    
    # Calculate the argument for the logarithm
    arg_plus = (omega_complex + q * v_F) / (omega_complex - q * v_F)
    
    # Calculate the Lindhard function
    D0 = prefactor * (1 - (omega_complex / (q * v_F)) * np.log(arg_plus))
    
    return D0


# Background: In a layered electron gas (LEG), the density-density correlation function D(l, l') describes the 
# response of the electron density in one layer to a perturbation in another layer. This function can be computed 
# using the Random Phase Approximation (RPA), which accounts for the screening effects due to the Coulomb interaction 
# between electrons in different layers. The Dyson equation is used to relate the bare correlation function D0 to the 
# full correlation function D, incorporating the self-energy effects from the Coulomb interaction. The matrix form of 
# the Dyson equation for the LEG involves the inversion of a matrix that includes the Coulomb interaction as a self-energy 
# term. The vacuum dielectric constant ε0 is used to normalize the Coulomb interaction, and the LEG dielectric constant 
# bg_eps modifies the interaction strength within the material. The matrix dimension N corresponds to the number of layers 
# considered in the calculation.


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
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # Initialize the interaction matrix V
    V = np.zeros((N, N), dtype=complex)

    # Fill the interaction matrix V using the form factor
    for l1 in range(N):
        for l2 in range(N):
            # Calculate the form factor f(q; l1, l2)
            delta_z = np.abs(l1 - l2) * d
            form_factor = np.exp(-q * delta_z) / bg_eps
            
            # Coulomb interaction term
            V[l1, l2] = (2 * np.pi * epsilon_0 / q) * form_factor

    # Identity matrix
    I = np.eye(N, dtype=complex)

    # Dyson equation: D = D0 + D0 * V * D
    # In matrix form: D = (I - D0 * V)^(-1) * D0
    D = np.linalg.inv(I - np.dot(D0, V)) @ D0

    return D



# Background: In a semi-infinite layered electron gas (LEG), the density-density correlation function D(l, l') 
# describes the response of the electron density in one layer to a perturbation in another layer. The Random Phase 
# Approximation (RPA) is used to account for the screening effects due to the Coulomb interaction between electrons 
# in different layers. The Dyson equation relates the bare correlation function D0 to the full correlation function D, 
# incorporating the self-energy effects from the Coulomb interaction. For a semi-infinite system, the matrix form of 
# the Dyson equation simplifies, and the explicit form of D(l, l') can be derived analytically. The vacuum dielectric 
# constant ε0 is used to normalize the Coulomb interaction, and the LEG dielectric constant bg_eps modifies the 
# interaction strength within the material. The form factor f(q; l1, l2) accounts for the exponential decay of the 
# interaction with layer separation.

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


    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1

    # Calculate the form factor f(q; l1, l2)
    delta_z = np.abs(l1 - l2) * d
    form_factor = np.exp(-q * delta_z) / bg_eps

    # Coulomb interaction term
    V_q = (2 * np.pi * epsilon_0 / q) * form_factor

    # Dyson equation for semi-infinite LEG: D = D0 / (1 - D0 * V_q)
    D_l = D0 / (1 - D0 * V_q)

    return D_l


from scicode.parse.parse import process_hdf5_to_tuple

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
