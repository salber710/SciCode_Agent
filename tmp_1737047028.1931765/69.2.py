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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('69.2', 3)
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
