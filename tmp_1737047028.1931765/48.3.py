import numpy as np
import scipy.interpolate as interpolate

# Background: 
# In electron scattering experiments, the momentum transfer vector Q is a crucial quantity that describes the change in momentum of the electron as it scatters off a sample. 
# The momentum transfer is given by Q = k_s - k_i, where k_i and k_s are the wave vectors of the incident and scattered electrons, respectively.
# The wave vector k is related to the electron's energy E by the relation k = sqrt(2mE)/ħ, where m is the electron mass and ħ is the reduced Planck's constant.
# In this problem, we are given the angles th and gamma, which describe the geometry of the scattering process. 
# The angle th is the angle between the incident electron and the sample surface normal, and gamma is the angle between the incident and scattered electron.
# The energy of the incident electron is E0, and omega is the energy loss during scattering.
# We need to calculate the in-plane momentum q and the out-of-plane momenta k_i^z and k_s^z.
# The in-plane momentum q is given by q = k_i^x - k_s^x, where k_i^x and k_s^x are the x-components of the incident and scattered wave vectors.
# The out-of-plane momenta k_i^z and k_s^z are the z-components of the incident and scattered wave vectors, respectively.
# The conversion from energy to wave vector involves the constants: electron mass m_e = 0.51099895 MeV/c^2 and Planck's constant hc = 1239.84193 eV nm.


def q_cal(th, gamma, E0, omega):
    '''Calculate the in-plane momentum q, and out-of-plane momenta of the incident and scattered electron k_i_z and k_s_z.
    Ensure that the signs of q, k_i_z and k_s_z are correctly represented.
    Input 
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    omega, energy loss, a list of float in the unit of eV
    Output
    Q: a tuple (q,k_i_z,k_s_z) in the unit of inverse angstrom, where q is in-plane momentum, 
       and k_i_z and k_s_z are out-of-plane momenta of the incident and scattered electron    
    '''
    
    # Constants
    m_e = 0.51099895 * 1e6  # Convert MeV/c^2 to eV/c^2
    hc = 1239.84193  # eV nm
    hbar = hc / (2 * np.pi)  # eV nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate initial and final energies
    E_i = E0
    E_s = E0 - np.array(omega)

    # Calculate wave vectors
    k_i = np.sqrt(2 * m_e * E_i) / hbar  # in nm^-1
    k_s = np.sqrt(2 * m_e * E_s) / hbar  # in nm^-1

    # Calculate components of wave vectors
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)
    k_i_x = k_i * np.sin(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)

    # Calculate in-plane momentum transfer q
    q = k_i_x - k_s_x

    # Convert from nm^-1 to angstrom^-1
    q_angstrom = q * 10
    k_i_z_angstrom = k_i_z * 10
    k_s_z_angstrom = k_s_z * 10

    return (q_angstrom, k_i_z_angstrom, k_s_z_angstrom)


# Background: 
# The Coulomb matrix element, V_eff, represents the effective interaction between charged particles in a medium.
# In the context of electron scattering, this interaction is expressed in reciprocal space at a momentum transfer vector 
# denoted as \tilde{Q} = (q, \kappa), where q is the in-plane momentum transfer and \kappa is the sum of the out-of-plane 
# momenta of the incident and scattered electrons, i.e., \kappa = k_i^z + k_s^z.
# The effective Coulomb interaction in the cgs system can be simplified by assuming 4\pi*e^2 = 1, where e is the elementary charge.
# This simplification allows us to focus on the geometric and energetic aspects of the scattering process without 
# explicitly considering the charge of the electron.
# The task is to calculate V_eff using the diffractometer angles and the electron energy, which involves computing 
# the components of \tilde{Q} and using them to determine the matrix element.


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
    m_e = 0.51099895 * 1e6  # Convert MeV/c^2 to eV/c^2
    hc = 1239.84193  # eV nm
    hbar = hc / (2 * np.pi)  # eV nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate initial and final energies
    E_i = E0
    E_s = E0 - np.array(omega)

    # Calculate wave vectors
    k_i = np.sqrt(2 * m_e * E_i) / hbar  # in nm^-1
    k_s = np.sqrt(2 * m_e * E_s) / hbar  # in nm^-1

    # Calculate components of wave vectors
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)
    k_i_x = k_i * np.sin(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)

    # Calculate in-plane momentum transfer q
    q = k_i_x - k_s_x

    # Calculate out-of-plane momentum sum kappa
    kappa = k_i_z + k_s_z

    # Convert from nm^-1 to angstrom^-1
    q_angstrom = q * 10
    kappa_angstrom = kappa * 10

    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1 / (q_angstrom**2 + kappa_angstrom**2)

    return V_eff



# Background: In electron scattering experiments, the measured cross section I(omega) is related to the density-density 
# correlation function S(omega) through the effective Coulomb interaction V_eff. In the low q limit, the cross section 
# is proportional to V_eff^2 * S(omega). The density-density correlation function S(omega) provides information about 
# the dynamic response of the system to external perturbations. To extract S(omega) from the measured cross section, 
# we need to account for the effective interaction V_eff, which depends on the scattering geometry and electron energy. 
# The task is to calculate S(omega) using the measured cross section I(omega), the diffractometer angles, and the 
# incident electron energy.

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
    m_e = 0.51099895 * 1e6  # Convert MeV/c^2 to eV/c^2
    hc = 1239.84193  # eV nm
    hbar = hc / (2 * np.pi)  # eV nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate initial and final energies
    E_i = E0
    E_s = E0 - np.array(omega)

    # Calculate wave vectors
    k_i = np.sqrt(2 * m_e * E_i) / hbar  # in nm^-1
    k_s = np.sqrt(2 * m_e * E_s) / hbar  # in nm^-1

    # Calculate components of wave vectors
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)
    k_i_x = k_i * np.sin(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)

    # Calculate in-plane momentum transfer q
    q = k_i_x - k_s_x

    # Calculate out-of-plane momentum sum kappa
    kappa = k_i_z + k_s_z

    # Convert from nm^-1 to angstrom^-1
    q_angstrom = q * 10
    kappa_angstrom = kappa * 10

    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1 / (q_angstrom**2 + kappa_angstrom**2)

    # Calculate S(omega) from I(omega) and V_eff
    S_omega = np.array(I) / (V_eff**2)

    return S_omega


from scicode.parse.parse import process_hdf5_to_tuple

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
