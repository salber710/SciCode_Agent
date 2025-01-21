import numpy as np
import itertools



# Background: 
# In quantum mechanics, the ground state energy of a particle in a 1D infinite square well is given by the formula:
# E_n = (n^2 * h^2) / (8 * m * L^2), where n is the principal quantum number (n=1 for ground state),
# h is the Planck constant, m is the particle's mass, and L is the width of the well.
# For an electron in a solid, we consider the effective mass, m_eff = m_r * m_e, where m_r is the relative effective mass
# and m_e is the free electron mass. The energy difference associated with this transition can be related to a photon
# wavelength λ by the equation E = h * c / λ, where c is the speed of light.
# To find the wavelength corresponding to the ground state energy, we first calculate the energy using the modified
# mass and then use the energy-wavelength relationship to solve for λ.

def ground_state_wavelength(L, mr):
    '''Given the width of an infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): Relative effective electron mass.
    
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    
    # Constants
    h = 6.626e-34  # Planck constant (Joule*second)
    c = 3e8        # Speed of light (meters/second)
    m_e = 9.109e-31 # Free electron mass (kg)
    
    # Convert L from nanometers to meters
    L_meters = L * 1e-9
    
    # Calculate the effective mass
    m_eff = mr * m_e
    
    # Calculate the ground state energy (n=1)
    E_1 = (h**2) / (8 * m_eff * L_meters**2)
    
    # Calculate the corresponding wavelength using E = h * c / λ
    lmbd_meters = h * c / E_1
    
    # Convert the wavelength from meters to nanometers
    lmbd = lmbd_meters * 1e9
    
    return lmbd

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('35.1', 3)
target = targets[0]

assert np.allclose(ground_state_wavelength(5,0.6), target)
target = targets[1]

assert np.allclose(ground_state_wavelength(10,0.6), target)
target = targets[2]

assert np.allclose(ground_state_wavelength(10,0.06), target)
