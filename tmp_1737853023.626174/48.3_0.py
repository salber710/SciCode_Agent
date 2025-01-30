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
    
    # Handle negative and excessive energy loss
    if any(omega_i < 0 for omega_i in omega):
        raise ValueError("Negative energy loss is not physically possible.")
    if any(omega_i > E0 for omega_i in omega):
        raise ValueError("Energy loss cannot be greater than the incident energy.")
    
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


# Background: In electron scattering experiments, the Coulomb matrix element is a measure of the effective interaction 
# between electrons in reciprocal space. It is calculated using the momentum transfer vector components derived from 
# diffractometer angles. The effective Coulomb interaction in reciprocal space is expressed at momentum 
# $\tilde{Q} = (q, \kappa)$, where $q$ is the in-plane momentum transfer and $\kappa = k_i^z + k_s^z$ is the sum of 
# the out-of-plane momenta of the incident and scattered electrons. In the cgs system, the Coulomb interaction is 
# often simplified by assuming $4\pi e^2 = 1$, where $e$ is the elementary charge. This simplification allows us to 
# focus on the geometric and energetic aspects of the interaction without the need for explicit charge values.


def MatELe(th, gamma, E0, omega):
    '''Calculate the Coulomb matrix element in the cgs system using diffractometer angles and the electron energy. 
    For simplicity, assume 4\\pi*e^2 = 1 where e is the elementary charge. 
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
    hc = 1239.84193  # Planck's constant times c in eV nm
    
    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)
    
    # Calculate the wave numbers for the incident and scattered electrons
    k_i = np.sqrt(2 * m_e * E0) / hc  # incident wave number in inverse nm
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hc  # scattered wave number in inverse nm
    
    # Handle cases where energy loss is greater than or equal to incident energy
    if any(E0 - omega_i <= 0 for omega_i in omega):
        raise ValueError("Energy loss cannot be greater than or equal to the incident energy.")
    
    # Calculate the in-plane momentum q
    q = k_i * np.sin(th_rad) - k_s * np.sin(th_rad + gamma_rad)
    
    # Calculate the out-of-plane momenta k_i^z and k_s^z
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)
    
    # Calculate kappa
    kappa = k_i_z + k_s_z
    
    # Convert from inverse nm to inverse angstrom
    q *= 10
    kappa *= 10
    
    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1 / (q**2 + kappa**2)
    
    return V_eff



# Background: In electron scattering experiments, the density-density correlation function, S(ω), is a key quantity 
# that describes how density fluctuations in a material respond to energy transfer ω. The measured cross section, I(ω), 
# is related to S(ω) through the effective Coulomb interaction, V_eff, in the low q limit. Specifically, the cross 
# section is proportional to V_eff^2 * S(ω). To extract S(ω) from experimental data, we need to account for the 
# effective interaction by dividing the measured cross section by V_eff^2. This process allows us to isolate the 
# intrinsic material response from the effects of the interaction.


def S_cal(omega, I, th, gamma, E0):
    '''Convert the experimental data to density-density correlation function, where σ_0 = 1 
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
    
    # Calculate kappa
    kappa = k_i_z + k_s_z
    
    # Convert from inverse nm to inverse angstrom
    q *= 10
    kappa *= 10
    
    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1 / (q**2 + kappa**2)
    
    # Calculate the density-density correlation function S(omega)
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
