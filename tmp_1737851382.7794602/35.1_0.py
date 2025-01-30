import numpy as np
import itertools



# Background: In quantum mechanics, the ground state energy of a particle in a 1D infinite square well is given by the formula:
# E = (h^2 * n^2) / (8 * m * L^2), where h is the Planck constant, n is the principal quantum number (n=1 for ground state),
# m is the effective mass of the particle, and L is the width of the well. The effective mass m is given by m_r * m_e,
# where m_r is the relative effective mass and m_e is the free electron mass. The energy E can be related to the wavelength
# of a photon by the equation E = h * c / λ, where c is the speed of light and λ is the wavelength. By equating these two
# expressions for energy, we can solve for the wavelength λ.

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
    m_e = 9.109e-31  # Free electron mass in kg

    # Convert L from nanometers to meters
    L_m = L * 1e-9

    # Calculate the effective mass
    m = mr * m_e

    # Calculate the ground state energy E
    E = (h**2) / (8 * m * L_m**2)

    # Calculate the wavelength λ using E = h * c / λ
    lmbd = h * c / E

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
