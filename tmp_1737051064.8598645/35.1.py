import numpy as np
import itertools



# Background: 
# In quantum mechanics, the ground state energy of a particle in a 1D infinite square well is given by the formula:
# E_n = (n^2 * h^2) / (8 * m * L^2), where n is the quantum number (n=1 for ground state), h is the Planck constant,
# m is the mass of the particle, and L is the width of the well. For the ground state, n=1.
# The effective mass m_r is used to account for the mass of the electron in a material, and it is given as a 
# multiple of the free electron mass m_0. Therefore, the effective mass m = m_r * m_0.
# The energy of a photon is related to its wavelength by the equation E = h * c / λ, where c is the speed of light.
# By equating the ground state energy to the photon energy, we can solve for the wavelength λ.

def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    # Constants
    h = 6.626e-34  # Planck constant in J*s
    c = 3e8        # Speed of light in m/s
    m0 = 9.109e-31 # Free electron mass in kg

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Calculate the effective mass
    m = mr * m0

    # Calculate the ground state energy E1
    E1 = (h**2) / (8 * m * L_m**2)

    # Calculate the wavelength λ corresponding to the ground state energy
    lmbd = (h * c) / E1

    # Convert the wavelength from meters to nanometers
    lmbd_nm = lmbd * 1e9

    return lmbd_nm


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('35.1', 3)
target = targets[0]

assert np.allclose(ground_state_wavelength(5,0.6), target)
target = targets[1]

assert np.allclose(ground_state_wavelength(10,0.6), target)
target = targets[2]

assert np.allclose(ground_state_wavelength(10,0.06), target)
