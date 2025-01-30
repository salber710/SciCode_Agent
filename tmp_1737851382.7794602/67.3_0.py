import numpy as np

# Background: In a layered electron gas (LEG) system, the Coulomb interaction between two electrons
# is influenced by the dielectric constant of the material and the spatial separation between the layers.
# The interaction potential in a 3D system is given by V(r) = e^2 / (4 * pi * epsilon_0 * epsilon * r),
# where epsilon is the dielectric constant. In a 2D system, the Fourier transform of the Coulomb potential
# is used to simplify calculations, resulting in V_q = e^2 / (2 * epsilon * q) for in-plane momentum q.
# The form factor f(q;z,z') accounts for the layered structure and the positions of the electrons in the layers.
# It modifies the interaction potential to reflect the discrete nature of the layers and their separation.


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
    # Calculate the z positions of the two electrons
    z1 = l1 * d
    z2 = l2 * d
    
    # Handle the case where the dielectric constant is zero to avoid division by zero
    if bg_eps == 0:
        return 0
    
    # Calculate the form factor using the exponential decay due to the separation in the z-direction
    form_factor = np.exp(-np.abs(q) * np.abs(z1 - z2) / bg_eps)
    
    # Correct the form factor for cases where the layers are the same
    if l1 == l2:
        form_factor = 1.0
    
    return form_factor


# Background: In a two-dimensional electron gas (2DEG), the density-density correlation function, also known as the 
# Lindhard function, describes how the electron density responds to external perturbations. At zero temperature (T=0),
# this function can be derived using the random phase approximation (RPA). The correlation function D^0(q, omega + i*gamma)
# is a complex function that depends on the in-plane momentum q, the energy omega, and a small imaginary part gamma 
# which ensures causality. The effective electron density n_eff, Fermi energy e_F, Fermi momentum k_F, and Fermi velocity 
# v_F are parameters that characterize the 2DEG. The Lindhard function is computed by integrating over the occupied 
# states in the Fermi sea, and it captures the collective excitations (plasmons) and single-particle excitations 
# (particle-hole pairs) in the system.


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
    
    # Calculate the Lindhard function for a 2DEG
    # The Lindhard function is given by:
    # D^0(q, omega + i*gamma) = (m / (2 * pi * hbar^2)) * (1 - (omega + i*gamma) / sqrt((omega + i*gamma)^2 - (v_F * q)^2))
    # where m is the effective mass of the electron, which can be related to the Fermi velocity and Fermi momentum.
    
    # Effective mass m can be derived from the Fermi velocity and Fermi momentum:
    # m = hbar * k_F / v_F
    if k_F == 0:
        raise ZeroDivisionError("Fermi momentum k_F cannot be zero.")
    m = k_F / v_F  # in units of meV * angstrom^2
    
    # Calculate the Lindhard function
    if q == 0:
        # Handle the case where q is zero to avoid division by zero in the square root
        term1 = omega_complex / np.sqrt(omega_complex**2)
    else:
        term1 = omega_complex / np.sqrt(omega_complex**2 - (v_F * q)**2)
    
    D0 = (m / (2 * np.pi)) * (1 - term1)
    
    return D0



# Background: In a layered electron gas (LEG), the density-density correlation function D(l, l') is computed using the 
# random phase approximation (RPA). The RPA accounts for the collective response of the electron system to external 
# perturbations by summing over an infinite series of bubble diagrams. The Dyson equation in matrix form is used to 
# compute the full correlation function D from the non-interacting correlation function D0 and the Coulomb interaction 
# between layers. The Dyson equation is given by D = D0 + D0 * V * D, where V is the Coulomb interaction matrix. 
# In matrix notation, this becomes D = (I - D0 * V)^(-1) * D0, where I is the identity matrix. The Coulomb interaction 
# between layers is modified by the form factor f(q; l, l') and the dielectric constant of the LEG.


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
    
    # Initialize the Coulomb interaction matrix V
    V = np.zeros((N, N), dtype=complex)
    
    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^-1 μm^-1
    
    # Calculate the Coulomb interaction matrix elements
    for l1 in range(N):
        for l2 in range(N):
            # Calculate the form factor f(q; l1, l2)
            z1 = l1 * d
            z2 = l2 * d
            form_factor = np.exp(-np.abs(q) * np.abs(z1 - z2) / bg_eps)
            if l1 == l2:
                form_factor = 1.0
            
            # Calculate the Coulomb interaction V_q
            V_q = 1 / (2 * bg_eps * q)  # in units of e^2 / (2 * epsilon * q)
            
            # Calculate the matrix element V(l1, l2)
            V[l1, l2] = V_q * form_factor / epsilon_0
    
    # Calculate the matrix (I - D0 * V)
    I = np.eye(N, dtype=complex)
    D0V = np.dot(D0, V)
    I_minus_D0V = I - D0V
    
    # Invert the matrix (I - D0 * V)
    I_minus_D0V_inv = np.linalg.inv(I_minus_D0V)
    
    # Calculate the full density-density correlation function D
    D = np.dot(I_minus_D0V_inv, D0)
    
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
