import numpy as np

# Background: 
# The density of states (DOS) effective mass is a concept in semiconductor physics 
# that describes how the energy levels are distributed in a semiconductor material.
# In a semiconductor like Al_xGa_{1-x}As, which is a mixture of Aluminum (Al) and Gallium Arsenide (GaAs),
# the effective mass of charge carriers (electrons and holes) can vary with the composition (x) of the material.
# The DOS effective mass is important for determining the electronic properties of the material, 
# such as carrier concentration and mobility.
# The relative effective mass (m_r) is given by the geometric mean of the effective masses of the 
# conduction band (electron) and the valence band (heavy and light holes).
# For the conduction band, the effective electron mass is given by m_e = (0.0637 + 0.083 * x) * m_0.
# In the valence band, the heavy hole and light hole masses are given by:
# m_hh = (0.50 + 0.29 * x) * m_0 and m_lh = (0.087 + 0.063 * x) * m_0.
# The effective mass for density of states (m_r) is often calculated for semiconductors as:
# m_r = ((m_hh^2 * m_lh)^(1/3)) / m0, where m0 is the electron rest mass.

def m_eff(x, m0):
    '''Calculates the effective mass of GaAlAs for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in GaAlAs.
    m0 (float): electron rest mass (can be reduced to 1 as default).
    Output:
    mr (float): Effective mass of GaAlAs.
    '''

    # Calculate effective electron mass
    m_e = (0.0637 + 0.083 * x) * m0
    
    # Calculate heavy hole mass
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Calculate light hole mass
    m_lh = (0.087 + 0.063 * x) * m0
    
    # Calculate the density of states effective mass (DOS effective mass)
    # Using the geometric mean for the valence band (heavy and light holes)
    m_v = (m_hh**2 * m_lh)**(1/3)
    
    # Calculate the relative effective mass m_r
    mr = m_v / m0  # Normalizing by the electron rest mass
    
    return mr



# Background: 
# The effective absorption coefficient (α) of a semiconductor material like Al_xGa_{1-x}As is a measure 
# of how much light is absorbed per unit distance as it passes through the material. 
# The absorption coefficient depends on several factors, including the material composition (x), 
# the wavelength of the incident light (λ), and intrinsic material properties.
# In semiconductors, the absorption process is related to the transition of electrons from the valence band 
# to the conduction band, which is influenced by the bandgap energy. 
# The effective absorption coefficient can be computed using the relation that involves the photon energy, 
# effective masses, and possibly other constants.
# For light with wavelength λ, the photon energy E can be calculated using E = hc/λ, where h is Planck's constant 
# and c is the speed of light. The absorption coefficient α can be affected by the effective mass of the material,
# and it is often modeled as proportional to the square root of the photon energy minus the bandgap energy (E_g),
# and the density of states effective mass (m_r).
# Here, we are asked to compute α as a function of λ, x, and a constant C, incorporating these physics concepts.

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
    e_charge = 1.6e-19  # Electron charge in coulombs
    c_light = 3e8  # Speed of light in vacuum in m/s
    h_planck = 6.63e-34  # Planck's constant in J·s
    
    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9
    
    # Calculate photon energy (E = hc / lambda)
    photon_energy = (h_planck * c_light) / lambda_m
    
    # Calculate the density of states effective mass using the previous function
    m_r = m_eff(x, m0=1)
    
    # Assuming a model where the effective absorption coefficient is proportional
    # to the square root of the photon energy times the effective mass
    # This is a simplification and can be adjusted based on more detailed semiconductor physics
    alpha_x = C * np.sqrt(photon_energy * m_r)
    
    return alpha_x

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.2', 3)
target = targets[0]

assert np.allclose(alpha_eff(800, 0.2, 1), target)
target = targets[1]

assert np.allclose(alpha_eff(980, 0, 1), target)
target = targets[2]

assert np.allclose(alpha_eff(700, 0.2, 1), target)
