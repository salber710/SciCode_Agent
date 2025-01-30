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



def MatELe(th, gamma, E0, omega):
    # Constants
    m_e = 511e3  # electron mass in eV/c^2
    c = 3e8  # speed of light in m/s
    hbar = 6.582119569e-16  # Reduced Planck's constant in eV*s

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate wave vectors in m^-1
    k_i = np.sqrt(2 * m_e * E0) / (hbar * c)  # initial wave vector
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / (hbar * c)  # scattered wave vector

    # Components of the wave vectors
    k_i_x = k_i * np.sin(th_rad)
    k_i_z = k_i * np.cos(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)
    k_s_z = k_s * np.cos(th_rad + gamma_rad)

    # In-plane momentum transfer q and out-of-plane momentum transfer kappa
    q = np.abs(k_i_x - k_s_x)  # in m^-1
    kappa = np.abs(k_i_z + k_s_z)  # in m^-1

    # Convert m^-1 to Å^-1
    q = q * 1e10
    kappa = kappa * 1e10

    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1 / (q**2 + kappa**2 + 1e-20)  # use a very small constant to prevent division by zero

    return V_eff



# Background: The density-density correlation function, S(omega), is a measure of how density fluctuations in a material
# are correlated at different energy losses, omega. In scattering experiments, the measured intensity I(omega) is related
# to the density-density correlation function through the scattering cross-section. In the low q limit, the cross-section
# is proportional to the square of the effective Coulomb interaction, V_eff^2, times S(omega). Therefore, to extract
# S(omega) from the measured intensity, we need to divide I(omega) by V_eff^2. The effective Coulomb interaction, V_eff,
# is calculated using the momentum transfer q and kappa, which are derived from the diffractometer angles and the
# incident and scattered electron energies.


def S_cal(omega, I, th, gamma, E0):
    '''Convert the experimental data to density-density correlation function, where \sigma_0 = 1 
    Input 
    omega, energy loss, a list of float in the unit of eV
    I, measured cross section from the detector in the unit of Hz, a list of float
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    Output
    S_omega: density-density correlation function, a list of float
    '''
    
    # Constants
    m_e = 511e3  # electron mass in eV/c^2
    c = 3e8  # speed of light in m/s
    hbar = 6.582119569e-16  # Reduced Planck's constant in eV*s

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate wave vectors in m^-1
    k_i = np.sqrt(2 * m_e * E0) / (hbar * c)  # initial wave vector
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / (hbar * c)  # scattered wave vector

    # Components of the wave vectors
    k_i_x = k_i * np.sin(th_rad)
    k_i_z = k_i * np.cos(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)
    k_s_z = k_s * np.cos(th_rad + gamma_rad)

    # In-plane momentum transfer q and out-of-plane momentum transfer kappa
    q = np.abs(k_i_x - k_s_x)  # in m^-1
    kappa = np.abs(k_i_z + k_s_z)  # in m^-1

    # Convert m^-1 to Å^-1
    q = q * 1e10
    kappa = kappa * 1e10

    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1 / (q**2 + kappa**2 + 1e-20)  # use a very small constant to prevent division by zero

    # Calculate S(omega) from I(omega) and V_eff
    S_omega = I / (V_eff**2)

    return S_omega


try:
    targets = process_hdf5_to_tuple('48.3', 3)
    target = targets[0]
    th = np.linspace(35.14,36.48,10)
    gamma = 70*np.ones(len(th))
    E0 = 50.0
    omega = np.linspace(-0.2,2.0,10)
    np.random.seed(2024)
    I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
    assert np.allclose(S_cal(omega,I,th,gamma,E0), target)

    target = targets[1]
    th = np.linspace(40.14,41.48,10)
    gamma = 70*np.ones(len(th))
    E0 = 50.0
    omega = np.linspace(-0.2,2.0,10)
    np.random.seed(2024)
    I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
    assert np.allclose(S_cal(omega,I,th,gamma,E0), target)

    target = targets[2]
    th = np.linspace(50.14,51.48,10)
    gamma = 70*np.ones(len(th))
    E0 = 50.0
    omega = np.linspace(-0.2,2.0,10)
    np.random.seed(2024)
    I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
    assert np.allclose(S_cal(omega,I,th,gamma,E0), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e