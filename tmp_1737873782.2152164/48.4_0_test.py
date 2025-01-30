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
    m_e = 0.51099895e6  # electron mass in eV/c^2
    hc = 1239.84193  # Planck's constant times speed of light in eV*nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate the wave numbers for the incident and scattered electrons
    # k_i and k_s are the magnitudes of the wave vectors
    k_i = np.sqrt(2 * m_e * E0) / hc
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hc

    # Calculate the components of the wave vectors
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)  # negative because it's in the -z direction
    k_i_x = k_i * np.sin(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)

    # Calculate the in-plane momentum transfer q
    q = k_s_x - k_i_x

    # Return the calculated values
    return (q, k_i_z, k_s_z)



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
    m_e = 0.51099895e6  # electron mass in eV/c^2
    hc = 1239.84193  # Planck's constant times speed of light in eV*nm

    # Convert angles from degrees to radians
    th_rad = np.radians(th)
    gamma_rad = np.radians(gamma)

    # Calculate the wave numbers for the incident and scattered electrons
    k_i = np.sqrt(2 * m_e * E0) / hc
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hc

    # Calculate the components of the wave vectors
    k_i_z = k_i * np.cos(th_rad)
    k_s_z = -k_s * np.cos(th_rad + gamma_rad)  # negative because it's in the -z direction
    k_i_x = k_i * np.sin(th_rad)
    k_s_x = k_s * np.sin(th_rad + gamma_rad)

    # Calculate the in-plane momentum transfer q
    q = k_s_x - k_i_x

    # Calculate kappa
    kappa = k_i_z + k_s_z

    # Calculate the effective Coulomb matrix element V_eff
    # For simplicity, assume 4*pi*e^2 = 1
    V_eff = 1 / (q**2 + kappa**2)

    # Return the calculated Coulomb matrix element
    return V_eff


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

    # Calculate the effective Coulomb matrix element V_eff using the MatELe function
    V_eff = MatELe(th, gamma, E0, omega)

    # Convert the measured cross section to density-density correlation function S(omega)
    # The cross section is proportional to V_eff^2 * S(omega), so we solve for S(omega)
    S_omega = [i / (v_eff**2) for i, v_eff in zip(I, V_eff)]

    # Return the calculated density-density correlation function
    return S_omega





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

    # Calculate the effective Coulomb matrix element V_eff using the MatELe function
    V_eff = MatELe(th, gamma, E0, omega)

    # Convert the measured cross section to density-density correlation function S(omega)
    S_omega = [i / (v_eff**2) for i, v_eff in zip(I, V_eff)]

    # Prepare to antisymmetrize S(omega)
    # Create an interpolation function for S_omega
    interp_func = interpolate.interp1d(omega, S_omega, fill_value=0, bounds_error=False)

    # Antisymmetrize S(omega) to obtain chi''(omega)
    chi = []
    for om in omega:
        S_plus = interp_func(om)
        S_minus = interp_func(-om)
        # Antisymmetrization: chi''(omega) = (S(omega) - S(-omega)) / 2
        chi_prime_prime = (S_plus - S_minus) / 2
        # Append negative of the computed chi''(omega)
        chi.append(-chi_prime_prime)

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