import numpy as np
import scipy.interpolate as interpolate

# Background: 
# In scattering experiments, understanding the momentum transfer is crucial. The momentum transfer vector Q is defined as 
# the difference between the scattered wave vector (k_s) and the incident wave vector (k_i). For an electron scattering 
# off a surface, the wave vectors can be decomposed into components parallel and perpendicular to the surface. The 
# in-plane momentum q is along the surface, while the out-of-plane momentum components k_i^z and k_s^z are perpendicular 
# to the surface. The angles th and gamma are used to determine these components. The energy of the electron is related 
# to its momentum by the de Broglie relation, where the wave vector magnitude |k| = sqrt(2mE) / ħ. The calculations 
# require converting angles from degrees to radians and using trigonometric functions to find the components of k vectors 
# in the x and z directions. The constants given are the electron mass m_e in MeV/c^2 and Planck's constant hc in eV nm, 
# which will be used to compute the momenta in the chosen units.


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
    
    # Convert th and gamma from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)
    
    # Calculate initial and scattered energies
    E0_eV = E0
    E_s = np.array(E0_eV) - np.array(omega)  # Energy of the scattered electron in eV
    
    # Calculate the wave numbers
    k_i = np.sqrt(2 * m_e * E0_eV) / hc  # Initial wave number
    k_s = np.sqrt(2 * m_e * E_s) / hc    # Scattered wave number
    
    # Calculate components of the wave vectors
    k_i_x = k_i * np.sin(th_rad)
    k_i_z = k_i * np.cos(th_rad)
    
    k_s_x = k_s * np.sin(th_rad + gamma_rad)  # th + gamma gives the angle for k_s_x
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)  # k_s_z is in the negative z direction
    
    # Calculate in-plane momentum transfer q
    q = k_s_x - k_i_x
    
    # Prepare the output tuple
    Q = (q, k_i_z, k_s_z)
    
    return Q



# Background: In scattering experiments, the Coulomb matrix element describes the effective interaction 
# between charged particles, such as electrons, in reciprocal space. This is crucial for understanding 
# electron scattering processes. The effective Coulomb interaction, V_eff, at momentum \tilde{Q} = (q, \kappa) 
# is calculated using the in-plane momentum transfer q and the combined out-of-plane momentum \kappa = k_i^z + k_s^z. 
# In the cgs unit system and for simplification, we assume 4\pi*e^2 = 1, where e is the elementary charge. 
# The V_eff is thus determined by the inverse of the square magnitude of \tilde{Q}. The units for V_eff 
# are the inverse of square angstroms.

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
    
    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)
    
    # Calculate initial and scattered energies
    E0_eV = E0  # Incident energy in eV
    E_s = np.array(E0_eV) - np.array(omega)  # Scattered energy in eV
    
    # Calculate wave numbers
    k_i = np.sqrt(2 * m_e * E0_eV) / hc  # Initial wave number
    k_s = np.sqrt(2 * m_e * E_s) / hc    # Scattered wave number
    
    # Calculate components of the wave vectors
    k_i_x = k_i * np.sin(th_rad)
    k_i_z = k_i * np.cos(th_rad)
    
    k_s_x = k_s * np.sin(th_rad + gamma_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)
    
    # Calculate in-plane momentum transfer q
    q = k_s_x - k_i_x
    
    # Calculate combined out-of-plane momentum \kappa
    kappa = k_i_z + k_s_z
    
    # Calculate the magnitude of \tilde{Q}
    Q_tilde_magnitude_squared = q**2 + kappa**2
    
    # Calculate the effective Coulomb interaction V_eff
    V_eff = 1 / Q_tilde_magnitude_squared
    
    return V_eff

from scicode.parse.parse import process_hdf5_to_tuple
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
