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
    hc = 1239.84193  # Planck's constant times speed of light in eV*nm

    # Convert to radians
    th_rad = np.radians(90 - np.array(th))
    gamma_rad = np.radians(np.array(gamma))

    # Calculate wave vectors
    k_i = np.sqrt(2 * m_e * E0) / hc  # initial wave vector magnitude in inverse nm
    k_s = np.sqrt(2 * m_e * (E0 - np.array(omega))) / hc  # scattered wave vector magnitude in inverse nm

    # Calculate components of wave vectors
    k_i_x = k_i * np.cos(th_rad)
    k_i_z = k_i * np.sin(th_rad)
    k_s_x = k_s * np.cos(th_rad + gamma_rad)
    k_s_z = -k_s * np.sin(th_rad + gamma_rad)

    # Calculate momentum transfer
    q = k_s_x - k_i_x

    # Convert from inverse nm to inverse Å (1 nm = 10 Å)
    q = q * 10
    k_i_z = k_i_z * 10
    k_s_z = k_s_z * 10

    # Return the result as a tuple
    Q = (q, k_i_z, k_s_z)
    return Q


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e