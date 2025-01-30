import numpy as np
import scipy.interpolate as interpolate



# Background: 
# In electron scattering experiments, the momentum transfer and the momenta of the incident and scattered electrons are key quantities.
# The momentum transfer vector Q is defined as the difference between the wave vectors of the scattered (k_s) and incident (k_i) electrons.
# The wave vector k is related to the energy E of the electron by the relation k = sqrt(2mE)/h, where m is the electron mass and h is Planck's constant.
# In this problem, we are given angles th and gamma, which are related to the geometry of the scattering process.
# The angle th is the angle between the incident electron and the sample surface normal, and gamma is the angle between the incident and scattered electrons.
# The in-plane momentum q is related to the x-components of the wave vectors, while the out-of-plane momenta k_i^z and k_s^z are related to the z-components.
# The conversion from angles to momentum components involves trigonometric functions, where the x-component is related to the cosine of the angle and the z-component to the sine.


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
    m_e = 0.51099895 * 1e6  # electron mass in eV/c^2
    hc = 1239.84193  # Planck's constant times c in eV nm
    
    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)
    
    # Calculate the wave numbers for the incident and scattered electrons
    k_i = np.sqrt(2 * m_e * E0) / hc  # incident wave number in inverse nm
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hc  # scattered wave number in inverse nm
    
    # Calculate the in-plane momentum q
    q = k_i * np.sin(th_rad) - k_s * np.sin(th_rad + gamma_rad)
    
    # Calculate the out-of-plane momenta k_i^z and k_s^z
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)
    
    # Convert from inverse nm to inverse angstrom
    q *= 10
    k_i_z *= 10
    k_s_z *= 10
    
    return (q, k_i_z, k_s_z)

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
