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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('48.1', 3)
target = targets[0]

th = np.linspace(35.14,36.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
assert np.allclose(q_cal(th,gamma,E0,omega), target)
target = targets[1]

th = np.linspace(40.14,41.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
assert np.allclose(q_cal(th,gamma,E0,omega), target)
target = targets[2]

th = np.linspace(50.14,51.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
assert np.allclose(q_cal(th,gamma,E0,omega), target)
