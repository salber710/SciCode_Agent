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
    
    # Constants in different units for variation
    m_e = 0.51099895 * 1e6  # electron mass in eV/c^2
    h_bar_c_nm = 197.3269788  # Reduced Planck's constant times speed of light in eV·nm

    # Conversion factor for calculating wave numbers
    conversion_factor = (2 * m_e)**0.5 / h_bar_c_nm

    # Convert angles to radians using list comprehensions
    th_rad = [(90 - angle) * (3.141592653589793 / 180) for angle in th]
    gamma_rad = [angle * (3.141592653589793 / 180) for angle in gamma]

    # Calculate the initial wave number
    k_i = conversion_factor * (E0**0.5)

    # Prepare the result list for V_eff
    V_eff_results = []

    for omega_val, th_r, gamma_r in zip(omega, th_rad, gamma_rad):
        # Calculate the scattered wave number
        k_s = conversion_factor * ((E0 - omega_val)**0.5)

        # Use trigonometric approximations for sine and cosine
        k_i_x = k_i * th_r  # Approximation: sin(th) ≈ th
        k_i_z = k_i * (1 - th_r**2 / 2)  # Approximation: cos(th) ≈ 1 - th^2/2

        k_s_x = k_s * (th_r + gamma_r)  # sin(th + gamma) ≈ th + gamma
        k_s_z = -k_s * (1 - (th_r + gamma_r)**2 / 2)  # cos(th + gamma) ≈ 1 - (th + gamma)^2/2

        # Calculate in-plane and out-of-plane momentum transfers
        q = k_s_x - k_i_x
        kappa = k_i_z + k_s_z

        # Calculate the magnitude of the momentum transfer vector
        Q_magnitude = (q**2 + kappa**2)**0.5

        # Compute the effective Coulomb matrix element
        V_eff = 1 / Q_magnitude  # Assuming 4\pi*e^2 = 1

        # Append the result to the list
        V_eff_results.append(V_eff)

    return V_eff_results


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