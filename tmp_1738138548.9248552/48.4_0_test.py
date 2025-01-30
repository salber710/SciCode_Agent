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



def S_cal(omega, I, th, gamma, E0):
    '''
    Convert the experimental data to density-density correlation function, where \sigma_0 = 1 
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

    # Momentum transfer q
    q = np.sqrt((k_i_x - k_s_x)**2 + (k_i_z - k_s_z)**2)  # total momentum transfer in m^-1

    # Convert m^-1 to Å^-1
    q = q * 1e10

    # Calculate the effective Coulomb matrix element V_eff using a reciprocal squared model
    V_eff = 1 / (q**2 + 1)**2  # Reciprocal squared decay of interaction with distance

    # Calculate S(omega) from I(omega) and V_eff
    S_omega = np.array(I) / V_eff**2

    return S_omega



# Background: The fluctuation-dissipation theorem relates the density-density correlation function S(omega) to the 
# imaginary part of the density response function, denoted as chi''(omega). To obtain chi''(omega), we need to 
# antisymmetrize S(omega) with respect to energy loss omega. This involves considering both positive and negative 
# energy values. The antisymmetrization process is defined as chi''(omega) = (S(omega) - S(-omega)) / 2. 
# If omega values fall outside the given range, S(omega) is assumed to be zero. This process helps in understanding 
# the response of a system to external perturbations and is crucial in many-body physics.

def chi_cal(omega, I, th, gamma, E0):
    '''Convert the density-density correlation function to the imaginary part of the density response function 
    by antisymmetrizing S(omega). Temperature is not required for this conversion.
    Input 
    omega, energy loss, a list of float in the unit of eV
    I, measured cross section from the detector in the unit of Hz, a list of float
    th, angle between the incident electron and the sample surface normal is 90-th, a list of float in the unit of degree
    gamma, angle between the incident and scattered electron, a list of float in the unit of degree
    E0, incident electron energy, float in the unit of eV
    Output
    chi: negative of the imaginary part of the density response function, a list of float
    '''



    # Calculate S(omega) using the provided S_cal function
    S_omega = S_cal(omega, I, th, gamma, E0)

    # Create an interpolation function for S(omega)
    S_interp = interpolate.interp1d(omega, S_omega, bounds_error=False, fill_value=0)

    # Antisymmetrize S(omega) to get chi''(omega)
    chi = []
    for w in omega:
        S_w = S_interp(w)
        S_neg_w = S_interp(-w)
        chi_w = (S_w - S_neg_w) / 2
        chi.append(-chi_w)  # Return the negative of the imaginary part

    return chi


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e