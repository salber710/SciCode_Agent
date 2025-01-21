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
# The absorption coefficient in semiconductors like Al_xGa_{1-x}As is a measure of how far light can penetrate into the material before being absorbed. 
# This coefficient depends on the electronic band structure of the material, which is influenced by the composition x. 
# The effective absorption coefficient, α_x, can be related to the energy of the incident photons, which is inversely proportional to the wavelength λ. 
# The energy E of a photon is given by E = h*c/λ, where h is the reduced Planck constant, and c is the speed of light. 
# The absorption typically scales with the density of states and the probability of electronic transitions, both of which can depend on the effective mass of carriers.
# The effective mass m_r from the DOS can influence the joint density of states and hence the absorption.
# For simplification, a constant C is introduced to represent other material and system-specific factors affecting absorption, and the result is scaled accordingly.


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
    e_charge = 1.6e-19  # Electron charge in Coulombs
    c = 3e8  # Speed of light in m/s
    h_bar = 1.055e-34  # Reduced Planck constant in J·s

    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9

    # Calculate the photon energy in joules
    E_photon = h_bar * c / lambda_m

    # Use m_eff function to get the effective mass (assuming m0 = 1)
    m_r = m_eff(x, m0=1)

    # The effective absorption coefficient α_x can be a function of the 
    # photon energy and the effective mass. This is a simplified model.
    # α_x is proportional to the energy of the photon and inversely proportional
    # to the effective mass.
    alpha_x = C * (E_photon / m_r)

    return alpha_x



# Background: 
# In semiconductor physics, the absorption coefficient is a measure of how much light of a 
# particular wavelength is absorbed per unit distance as it travels through a material. 
# In the context of Al_xGa_{1-x}As, the absorption coefficient changes with the composition (x) 
# and the wavelength of light. The function alpha_eff(lambda_i, x, C) provides a model to compute 
# the effective absorption coefficient for a given composition and wavelength.
# To determine the actual absorption coefficient of Al_xGa_{1-x}As, we need to consider the 
# absorption coefficient of GaAs at a reference wavelength, lambda_0, denoted as alpha_0.
# The goal is to compute the absorption coefficient of Al_xGa_{1-x}As at any given wavelength 
# lambda_i, normalized by the reference absorption coefficient alpha_0 at lambda_0 for pure GaAs.
# This is achieved by scaling the effective absorption coefficient computed by alpha_eff with alpha_0.


def alpha(lambda_i, x, lambda0, alpha0):
    '''Computes the absorption coefficient for given wavelength and Al composition,
    normalized by the absorption coefficient at a reference wavelength for pure GaAs.
    Input:
    lambda_i (float): Wavelength of the incident light (nm).
    x (float): Aluminum composition in the AlxGa1-xAs alloy.
    lambda0 (float): Reference wavelength (nm) for pure GaAs (x=0).
    alpha0 (float): Absorption coefficient at the reference wavelength for pure GaAs.
    Output:
    alpha_final (float): Normalized absorption coefficient in m^-1.
    '''
    
    # Calculate the effective absorption coefficient for the given lambda_i and x
    alpha_eff_i = alpha_eff(lambda_i, x, C=1)
    
    # Calculate the effective absorption coefficient for the reference wavelength lambda0 and x=0
    alpha_eff_0 = alpha_eff(lambda0, 0, C=1)
    
    # Normalize the effective absorption coefficient by the absorption coefficient at lambda0
    alpha_final = alpha_eff_i * (alpha0 / alpha_eff_0)
    
    return alpha_final

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.3', 4)
target = targets[0]

assert (alpha(850, 0.2, 850, 9000) == 0) == target
target = targets[1]

assert np.allclose(alpha(800, 0.1, 850, 8000), target)
target = targets[2]

assert np.allclose(alpha(700, 0.2, 850, 9000), target)
target = targets[3]

assert np.allclose(alpha(700, 0.1, 850, 9000), target)
