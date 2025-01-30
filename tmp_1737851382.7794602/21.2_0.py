import numpy as np

# Background: The density of states (DOS) effective mass is a concept used in semiconductor physics to describe the 
# effective mass of charge carriers (electrons and holes) in a semiconductor material. It is a measure of how the 
# density of available electronic states changes with energy. For a compound semiconductor like Al_xGa_{1-x}As, 
# the effective masses of electrons and holes depend on the composition, specifically the aluminum mole fraction x. 
# The effective mass of electrons (m_e), light holes (m_lh), and heavy holes (m_hh) are given as functions of x. 
# The relative effective mass m_r is calculated using the formula:
# m_r = ((m_e * (m_hh^2/3) * (m_lh^1/3))^(1/3)) / m0
# where m0 is the electron rest mass. This formula accounts for the contributions of both heavy and light holes 
# to the density of states.


def m_eff(x, m0):
    '''Calculates the effective mass of GaAlAs for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in GaAlAs.
    m0 (float): electron rest mass (can be reduced to 1 as default).
    Output:
    mr (float): Effective mass of GaAlAs.
    '''
    if not isinstance(x, (int, float)):
        raise TypeError("Aluminum mole fraction x must be an integer or float.")
    if not isinstance(m0, (int, float)):
        raise TypeError("Electron rest mass m0 must be an integer or float.")
    if x < 0 or x > 1:
        raise ValueError("Aluminum mole fraction x must be between 0 and 1 inclusive.")
    
    # Calculate the effective masses based on the given formulas
    m_e = (0.0637 + 0.083 * x) * m0
    m_lh = (0.087 + 0.063 * x) * m0
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Calculate the relative effective mass m_r
    mr = ((m_e * (m_hh**(2/3)) * (m_lh**(1/3)))**(1/3))
    
    return mr



# Background: The effective absorption coefficient, α, in semiconductors is a measure of how much light is absorbed per unit distance as it travels through the material. It is influenced by the material's band structure and the interaction of photons with electrons and holes. For a compound semiconductor like Al_xGa_{1-x}As, the absorption coefficient depends on the composition x, the wavelength of the incident light λ, and other material properties. The effective mass of charge carriers, which can be calculated using the m_eff function, plays a role in determining the absorption properties. The absorption coefficient can be influenced by the density of states and the transition probabilities between energy levels. Constants such as the electron charge, speed of light, and Planck's constant are used in the calculations to relate the energy of photons to their wavelength.

def alpha_eff(lambda_i, x, C=1):
    '''Calculates the effective absorption coefficient of AlxGa1-xAs.
    Input:
    lambda_i (float): Wavelength of the incident light (nm).
    x (float): Aluminum composition in the AlxGa1-xAs alloy.
    C (float): Optional scaling factor for the absorption coefficient. Default is 1.
    Returns:
    Output (float): Effective absorption coefficient in m^-1.
    '''


    # Constants
    e = 1.6e-19  # Electron charge in Coulombs
    c = 3e8      # Speed of light in m/s
    h_bar = 1.05e-34  # Reduced Planck's constant in J·s

    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9

    # Calculate the energy of the photon
    E_photon = (h_bar * c) / lambda_m

    # Calculate the effective mass using the given function
    m_r = m_eff(x, m0=1)

    # Calculate the effective absorption coefficient
    alpha_x = C * (e**2) * m_r * E_photon / (h_bar**2 * c)

    return alpha_x

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.2', 3)
target = targets[0]

assert np.allclose(alpha_eff(800, 0.2, 1), target)
target = targets[1]

assert np.allclose(alpha_eff(980, 0, 1), target)
target = targets[2]

assert np.allclose(alpha_eff(700, 0.2, 1), target)
