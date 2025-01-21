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


# Background: The Coulomb matrix element describes the effective interaction between charged particles in a scattering experiment. 
# In this context, we are considering the interaction in reciprocal space, which is characterized by momentum transfer vectors. 
# The effective Coulomb interaction is simplified by assuming a constant factor 4πe² = 1, where e is the elementary charge. 
# The momentum transfer vector is represented as (q, κ), where q is the in-plane momentum transfer and κ is the sum of the 
# out-of-plane momenta of the incident and scattered electrons, k_i^z and k_s^z respectively. 
# The matrix element V_eff is calculated using these components and is expressed in units of inverse square angstroms.


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

    # Calculate kappa
    kappa = k_i_z + k_s_z

    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1.0 / (q**2 + kappa**2)

    return V_eff.tolist()


# Background: In scattering experiments, the density-density correlation function S(ω) provides important insights into 
# the dynamic properties of the material under study. In the low q limit, where the momentum transfer is small, the 
# measured cross section I(ω) is directly proportional to the square of the effective Coulomb matrix element V_eff and 
# the density-density correlation function S(ω). The relationship can be mathematically expressed as I(ω) ∝ V_eff^2 S(ω). 
# To extract S(ω) from the measured cross section data, we can rearrange this relationship to S(ω) = I(ω) / V_eff^2, 
# assuming a proportionality constant σ_0 = 1. This requires calculating the effective Coulomb matrix element for each 
# set of experimental conditions.


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

    # Calculate kappa
    kappa = k_i_z + k_s_z

    # Calculate the effective Coulomb matrix element V_eff
    V_eff = 1.0 / (q**2 + kappa**2)

    return V_eff.tolist()

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

    # Calculate the effective Coulomb matrix elements V_eff
    V_eff = MatELe(th, gamma, E0, omega)
    
    # Convert I(omega) to S(omega) using the relationship S(omega) = I(omega) / V_eff^2
    S_omega = np.array(I) / np.array(V_eff)**2
    
    return S_omega.tolist()



# Background: The fluctuation-dissipation theorem provides a connection between the response of a system to 
# an external perturbation and its internal fluctuations. In the context of this problem, the density-density 
# correlation function S(ω) is related to the imaginary part of the density response function, χ''(ω). 
# The antisymmetrization process is used to extract χ''(ω) from S(ω). The function χ''(ω) should be antisymmetric 
# with respect to ω, meaning χ''(-ω) = -χ''(ω). In order to achieve this, for each positive ω, we calculate 
# the antisymmetrized value as (S(ω) - S(-ω)) / 2. If ω falls outside the given range of energies, S(ω) is set 
# to zero for these calculations.



def chi_cal(omega, I, th, gamma, E0):
    '''Convert the density-density correlation function to the imaginary part of the density response function 
    by antisymmetrizing S(\omega). Temperature is not required for this conversion.
    Input 
    omega, energy loss, a list of float in the unit of eV
    I, measured cross section from the detector in the unit of Hz, a list of float
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    Output
    chi: negative of the imaginary part of the density response function, a list of float
    '''
    
    # Calculate S(omega) using the previous function S_cal
    S_omega = S_cal(omega, I, th, gamma, E0)

    # Convert omega and S_omega to numpy arrays for easier manipulation
    omega = np.array(omega)
    S_omega = np.array(S_omega)

    # Create an interpolating function for S(omega)
    S_interp = interpolate.interp1d(omega, S_omega, bounds_error=False, fill_value=0)

    # Initialize chi array with zeros
    chi = np.zeros_like(omega)

    # Antisymmetrize S(omega) to obtain chi''(omega)
    for i, w in enumerate(omega):
        S_w = S_interp(w)
        S_neg_w = S_interp(-w)
        chi[i] = (S_w - S_neg_w) / 2

    # Return the negative of the imaginary part of the density response function
    return -chi.tolist()

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('48.4', 4)
target = targets[0]

th = np.linspace(35.14,36.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
np.random.seed(2024)
I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
assert np.allclose(chi_cal(omega,I,th,gamma,E0), target)
target = targets[1]

th = np.linspace(40.14,41.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
np.random.seed(2024)
I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
assert np.allclose(chi_cal(omega,I,th,gamma,E0), target)
target = targets[2]

th = np.linspace(50.14,51.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
np.random.seed(2024)
I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
assert np.allclose(chi_cal(omega,I,th,gamma,E0), target)
target = targets[3]

th = np.linspace(55.14,56.48,10)
gamma = 70*np.ones(len(th))
E0 = 50.0
omega = np.linspace(-0.2,2.0,10)
np.random.seed(2024)
I = np.hstack((np.random.randint(0, 10, size=1)/3,np.random.randint(10, 101, size=9)/3))
omega_res = 0.1
chi = chi_cal(omega,I,th,gamma,E0)
assert ((chi[omega>omega_res]>0).all()) == target
