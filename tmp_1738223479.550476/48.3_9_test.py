from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import scipy.interpolate as interpolate

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
    hc = 1239.84193  # Planck's constant times speed of light in eV·nm

    # Conversion factor for calculating wave numbers
    factor = np.sqrt(2 * m_e) / hc

    # Convert angles from degrees to radians
    th_rad = np.deg2rad(np.subtract(90, th))
    gamma_rad = np.deg2rad(gamma)

    # Calculate the initial wave number
    k_i = factor * np.sqrt(E0)

    # Prepare result containers
    q_values = []
    k_i_z_values = []
    k_s_z_values = []

    # Iterate over each value of energy loss
    for omega_val, th_r, gamma_r in zip(omega, th_rad, gamma_rad):
        # Calculate the scattered wave number
        k_s = factor * np.sqrt(E0 - omega_val)

        # Calculate components of the wave vectors
        k_i_x = k_i * np.sin(th_r)
        k_i_z = k_i * np.cos(th_r)

        k_s_x = k_s * np.sin(th_r + gamma_r)
        k_s_z = -k_s * np.cos(th_r + gamma_r)

        # Calculate in-plane momentum transfer q
        q = k_s_x - k_i_x

        # Append the results to the lists
        q_values.append(q)
        k_i_z_values.append(k_i_z)
        k_s_z_values.append(k_s_z)

    # Return the results as a tuple of lists
    return (q_values, k_i_z_values, k_s_z_values)


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
    electron_mass = 0.51099895 * 1e6  # electron mass in eV/c^2
    planck_c_ev_pm = 1239.84193 * 10  # Planck's constant times speed of light in eV·pm

    # Conversion factor for calculating wave numbers
    conversion_factor = (2 * electron_mass)**0.5 / planck_c_ev_pm

    # Convert angles from degrees to radians
    th_rad = [(90 - angle) * (3.141592653589793 / 180) for angle in th]
    gamma_rad = [angle * (3.141592653589793 / 180) for angle in gamma]

    # Calculate the initial wave number
    k_initial = conversion_factor * (E0**0.5)

    # Prepare result container
    V_eff_results = []

    # Iterate over each value of energy loss
    for omega_single, th_single, gamma_single in zip(omega, th_rad, gamma_rad):
        # Calculate the scattered wave number
        k_scattered = conversion_factor * ((E0 - omega_single)**0.5)

        # Calculate components of the wave vectors using different approach
        k_i_x = k_initial * (th_single)  # Using small angle approximation
        k_i_z = k_initial * (1 - th_single**2 / 2)  # Small angle approximation for cosine

        k_s_x = k_scattered * (th_single + gamma_single)
        k_s_z = -k_scattered * (1 - (th_single + gamma_single)**2 / 2)

        # Calculate in-plane momentum transfer q
        q = k_s_x - k_i_x

        # Calculate out-of-plane momentum sum kappa
        kappa = k_i_z + k_s_z

        # Calculate the magnitude of Q
        Q_magnitude = (q**2 + kappa**2)**0.5
        
        # Calculate the effective Coulomb matrix element V_eff
        V_eff = 1 / Q_magnitude  # Assuming 4\pi*e^2 = 1

        # Append result
        V_eff_results.append(V_eff)

    return V_eff_results



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
    electron_mass_eV_c2 = 0.51099895e6  # electron mass in eV/c^2
    h_c_eV_pm = 1239.84193e1  # Planck's constant times speed of light in eV·pm

    # Convert angles from degrees to radians
    th_rad = np.pi * (90 - np.array(th)) / 180.0
    gamma_rad = np.pi * np.array(gamma) / 180.0

    # Function to compute wave vector magnitude
    def compute_k(energy):
        return np.sqrt(2 * electron_mass_eV_c2 * energy) / h_c_eV_pm

    # Initial wave vector magnitude
    k_initial = compute_k(E0)

    # Convert inputs to numpy arrays for vectorized computation
    omega = np.array(omega)
    I = np.array(I)

    # Scattered wave vector magnitudes
    k_scattered = compute_k(E0 - omega)

    # Alternate method: calculate wave vector components using vector cross products
    k_i_x = k_initial * np.cos(th_rad + gamma_rad)  # cosine component mixed for variation
    k_i_z = k_initial * np.sin(th_rad + gamma_rad)

    k_s_x = k_scattered * np.cos(th_rad)  # using cos for scattered as a variation
    k_s_z = k_scattered * np.sin(th_rad)

    # Calculate momentum transfer vector q
    q_x = k_s_x - k_i_x
    q_z = k_s_z + k_i_z

    # Compute magnitude of the momentum transfer Q using an alternative method
    Q_magnitude = np.hypot(q_x, q_z)

    # Compute the effective Coulomb matrix element V_eff
    V_eff = 1 / Q_magnitude  # Assuming 4πe² = 1 for simplicity

    # Compute the density-density correlation function S(omega)
    S_omega = I / (V_eff**2)

    return S_omega.tolist()


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