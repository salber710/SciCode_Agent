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
# The absorption coefficient, α, is a measure of how much light is absorbed per unit distance in a material.
# For semiconductors like Al_xGa_{1-x}As, the absorption coefficient is influenced by the material's 
# band structure and the interaction of photons with the electronic states of the material.
# The effective absorption coefficient, α_x, depends on several factors, including the wavelength of the
# incident light, λ, and the composition of the material, x. The absorption process can be influenced by 
# the transition of electrons from the valence band to the conduction band, which is often modeled using
# parameters such as the effective mass and energy band gap.
# The effective mass of the charge carriers affects the density of states and the probability of absorption.
# In this context, we are considering a simplified model where the absorption coefficient is proportional to
# an effective density of states and other constants. The function m_eff(x) has been previously defined 
# to compute the effective mass for the density of states, which plays a role in determining α_x.
# Here, we calculate α_x using these parameters and a scaling factor C.


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
    e = 1.6e-19  # Electron charge in C
    c = 3e8  # Speed of light in m/s
    h_bar = 1.05e-34  # Reduced Planck constant in J·s

    # Convert wavelength from nm to m
    lambda_m = lambda_i * 1e-9

    # Calculate the photon energy E = h_bar * c / lambda
    E_photon = h_bar * c / lambda_m

    # Calculate the effective mass using the given function m_eff
    # Assume m0=1 as per the previous step
    m_r = m_eff(x, m0=1)

    # Effective density of states related factor, assumed to be proportional to m_r
    dos_effective = m_r

    # Calculate the absorption coefficient alpha_x
    # Assuming alpha_x is proportional to dos_effective and photon energy
    # The actual formula would depend on the detailed material properties and physics
    alpha_x = C * dos_effective * E_photon / e

    return alpha_x

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.2', 3)
target = targets[0]

assert np.allclose(alpha_eff(800, 0.2, 1), target)
target = targets[1]

assert np.allclose(alpha_eff(980, 0, 1), target)
target = targets[2]

assert np.allclose(alpha_eff(700, 0.2, 1), target)
