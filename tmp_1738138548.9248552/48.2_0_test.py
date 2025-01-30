from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy.interpolate as interpolate


def q_cal(th, gamma, E0, omega):
    m_e = 0.51099895 * 1e6  # electron mass in eV/c^2
    hc = 1239.84193  # Planck's constant times speed of light in eV nm
    hbar_c = hc / (2 * np.pi)  # Reduced Planck's constant times c in eV nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate wave vectors
    k_i = np.sqrt(2 * m_e * E0) / hbar_c  # initial wave vector in nm^-1
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hbar_c  # scattered wave vector in nm^-1

    # Components of the wave vectors
    k_i_x = k_i * np.sin(th_rad)
    k_i_z = k_i * np.cos(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)

    # In-plane momentum transfer q
    q = k_i_x - k_s_x  # in nm^-1

    # Convert nm^-1 to Å^-1
    q = q * 10
    k_i_z = k_i_z * 10
    k_s_z = k_s_z * 10

    # Return the tuple
    return (q, k_i_z, k_s_z)



# Background: The Coulomb matrix element, V_eff, is a measure of the effective interaction between charged particles in a medium. 
# In this context, it is calculated in reciprocal space at a momentum transfer vector, denoted as (q, κ), where q is the in-plane 
# momentum transfer and κ is the sum of the out-of-plane momenta of the incident and scattered electrons, k_i^z and k_s^z, respectively. 
# The effective Coulomb interaction in the cgs system is often simplified by assuming 4πe^2 = 1, where e is the elementary charge. 
# This simplification allows us to focus on the geometric and energetic aspects of the interaction without the need for explicit 
# charge values. The result, V_eff, is expressed in units of inverse square angstroms, which is a common unit for potential energy 
# in reciprocal space.


def MatELe(th, gamma, E0, omega):
    '''Calculate the Coulomb matrix element in the cgs system using diffractometer angles and the electron energy. 
    For simplicity, assume 4\pi*e^2 = 1 where e is the elementary charge. 
    Input 
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    omega, energy loss, a list of float in the unit of eV
    Output
    V_eff: matrix element in the unit of the inverse of square angstrom, a list of float
    '''
    
    # Constants
    m_e = 0.51099895 * 1e6  # electron mass in eV/c^2
    hc = 1239.84193  # Planck's constant times speed of light in eV nm
    hbar_c = hc / (2 * np.pi)  # Reduced Planck's constant times c in eV nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate wave vectors
    k_i = np.sqrt(2 * m_e * E0) / hbar_c  # initial wave vector in nm^-1
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hbar_c  # scattered wave vector in nm^-1

    # Components of the wave vectors
    k_i_x = k_i * np.sin(th_rad)
    k_i_z = k_i * np.cos(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)

    # In-plane momentum transfer q
    q = k_i_x - k_s_x  # in nm^-1

    # Convert nm^-1 to Å^-1
    q = q * 10
    k_i_z = k_i_z * 10
    k_s_z = k_s_z * 10

    # Calculate κ
    kappa = k_i_z + k_s_z

    # Calculate the effective Coulomb matrix element V_eff
    # V_eff = 1 / (q^2 + kappa^2)
    V_eff = 1 / (q**2 + kappa**2)

    return V_eff


try:
    targets = process_hdf5_to_tuple('48.2', 3)
    target = targets[0]
    th = np.linspace(35.14,36.48,10)
    gamma = 70*np.ones(len(th))
    E0 = 50.0
    omega = np.linspace(-0.2,2.0,10)
    assert np.allclose(MatELe(th,gamma,E0,omega), target)

    target = targets[1]
    th = np.linspace(40.14,41.48,10)
    gamma = 70*np.ones(len(th))
    E0 = 50.0
    omega = np.linspace(-0.2,2.0,10)
    assert np.allclose(MatELe(th,gamma,E0,omega), target)

    target = targets[2]
    th = np.linspace(50.14,51.48,10)
    gamma = 70*np.ones(len(th))
    E0 = 50.0
    omega = np.linspace(-0.2,2.0,10)
    assert np.allclose(MatELe(th,gamma,E0,omega), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e