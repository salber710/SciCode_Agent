import numpy as np

# Background: In a layered electron gas (LEG) system, the Coulomb interaction between two electrons
# is influenced by the dielectric constant of the material and the spatial separation between the
# electron layers. The interaction potential in real space is given by the Coulomb potential, which
# in a 2D system is modified by the presence of the dielectric medium. The Fourier transform of this
# interaction with respect to the in-plane coordinates results in a form factor that depends on the
# in-plane momentum transfer q and the positions of the electrons in the layers, denoted by l1 and l2.
# The form factor f(q;z,z') is a crucial component in determining the effective interaction in the
# momentum space, and it accounts for the layered structure of the electron gas. The form factor
# typically involves exponential terms that decay with the separation between the layers, modulated
# by the dielectric constant.


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
    # Calculate the separation between the layers in the z-direction
    z1 = l1 * d
    z2 = l2 * d
    delta_z = np.abs(z1 - z2)
    
    # Calculate the form factor using the exponential decay with respect to the layer separation
    form_factor = np.exp(-q * delta_z / bg_eps)
    
    return form_factor


# Background: The density-density correlation function, also known as the Lindhard function, is a key
# quantity in many-body physics, particularly in the study of electron gases. At zero temperature (T=0),
# the time-ordered density-density correlation function for a two-dimensional electron gas (2DEG) can be
# derived using the random phase approximation (RPA). The function D^0(q, ω + iγ) describes how the 
# electron density responds to external perturbations characterized by momentum transfer q and energy 
# transfer ω. The imaginary part γ is introduced to account for damping or broadening effects. The 
# correlation function is computed by integrating over the occupied states in the Fermi sea, and it 
# depends on parameters such as the Fermi energy (e_F), Fermi momentum (k_F), and Fermi velocity (v_F).


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
    
    # Convert omega and gamma to complex frequency
    omega_complex = omega + 1j * gamma
    
    # Calculate the dimensionless variables
    x = q / (2 * k_F)
    y = omega_complex / (q * v_F)
    
    # Initialize the density-density correlation function
    D0 = np.zeros_like(q, dtype=complex)
    
    # Calculate the density-density correlation function
    for i in range(len(q)):
        if x[i] < 1:
            D0[i] = n_eff / (2 * np.pi * e_F) * (1 - x[i]**2) * (1 - y[i]**2)**0.5
        else:
            D0[i] = n_eff / (2 * np.pi * e_F) * (1 - y[i]**2)**0.5
    
    return D0



# Background: In a layered electron gas (LEG), the density-density correlation function D(l, l') is 
# computed using the Random Phase Approximation (RPA). The RPA accounts for the collective response 
# of the electron system to external perturbations by summing over an infinite series of bubble diagrams. 
# The Dyson equation in matrix form is used to compute the full correlation function D(l, l') from the 
# non-interacting correlation function D0 and the Coulomb interaction. The Coulomb interaction between 
# layers is treated as a self-energy term. The dielectric constant of the vacuum, ε0, is used to normalize 
# the interaction. The matrix form of the Dyson equation is given by D = D0 + D0 * V * D, where V is the 
# Coulomb interaction matrix. This can be solved using matrix inversion: D = (I - D0 * V)^(-1) * D0.


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
    
    # Initialize the Coulomb interaction matrix V
    V = np.zeros((N, N), dtype=complex)
    
    # Fill the Coulomb interaction matrix V
    for l1 in range(N):
        for l2 in range(N):
            # Calculate the form factor for the interaction between layers l1 and l2
            form_factor = np.exp(-q * np.abs(l1 - l2) * d / bg_eps)
            # Coulomb interaction in 2D
            V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)
            # Interaction matrix element
            V[l1, l2] = V_q * form_factor
    
    # Convert D0 to a diagonal matrix
    D0_matrix = np.diag(D0)
    
    # Calculate the matrix (I - D0 * V)
    I = np.eye(N, dtype=complex)
    I_minus_D0V = I - np.dot(D0_matrix, V)
    
    # Invert the matrix (I - D0 * V)
    I_minus_D0V_inv = np.linalg.inv(I_minus_D0V)
    
    # Calculate the full density-density correlation function matrix D
    D = np.dot(I_minus_D0V_inv, D0_matrix)
    
    return D


from scicode.parse.parse import process_hdf5_to_tuple

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
