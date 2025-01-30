from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def m_eff(x, m0):
    '''Calculates the effective mass of GaAlAs for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in GaAlAs.
    m0 (float): electron rest mass (can be reduced to 1 as default).
    Output:
    mr (float): Effective mass of GaAlAs.
    '''
    # Calculate the effective masses based on the given formulas
    m_e = (0.0637 + 0.083 * x) * m0
    m_lh = (0.087 + 0.063 * x) * m0
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Calculate the DOS effective mass for holes using a contraharmonic mean
    m_h = (m_lh**2 + m_hh**2) / (m_lh + m_hh)
    
    # The relative effective mass m_r is calculated using a quadratic mean
    m_r = ((m_e**2 + m_h**2) / 2)**0.5
    
    return m_r



# Background: 
# The absorption coefficient, α, of a semiconductor material such as AlxGa1-xAs is a measure of how much light of a specific wavelength is absorbed per unit distance as it travels through the material. 
# The effective absorption coefficient can be influenced by the electronic band structure of the material, which in turn is affected by the effective mass of the charge carriers.
# The absorption coefficient is often related to the photon energy which can be calculated using the relationship: E = h*c/λ, where E is the photon energy, h is the reduced Planck's constant, c is the speed of light, and λ is the wavelength.
# In this scenario, α is calculated as a function of the wavelength λ and aluminum composition x, scaled by a constant C which could represent other material-specific factors.
# We assume m_eff(x, m0=1) is a function defined earlier that calculates the density of states effective mass for the material based on the aluminum composition x.
# The effective absorption coefficient can then be approximated as proportional to a function of the photon energy and the effective mass, typically involving the square root of the energy term.

def alpha_eff(lambda_i, x, C):
    '''Calculates the effective absorption coefficient of AlxGa1-xAs.
    Input:
    lambda_i (float): Wavelength of the incident light (nm).
    x (float): Aluminum composition in the AlxGa1-xAs alloy.
    C (float): Optional scaling factor for the absorption coefficient. Default is 1.
    Returns:
    Output (float): Effective absorption coefficient in m^-1.
    '''
    # Constants
    h = 1.0545718e-34  # Reduced Planck constant in J·s
    c = 3e8            # Speed of light in m/s
    e = 1.60217663e-19 # Electron charge in C

    # Convert wavelength from nm to m
    lambda_m = lambda_i * 1e-9
    
    # Calculate photon energy (E = h*c/λ)
    E_photon = h * c / lambda_m
    
    # Use the effective mass function m_eff to get the effective mass
    m_r = m_eff(x, m0=1)
    
    # Calculate the effective absorption coefficient
    # Assuming a proportionality relation: α ~ C * sqrt(E_photon) / m_r
    # This is a simplified form of calculating α where the absorption is proportional to the energy.
    alpha_x = C * (E_photon**0.5) / m_r
    
    return alpha_x


try:
    targets = process_hdf5_to_tuple('21.2', 3)
    target = targets[0]
    assert np.allclose(alpha_eff(800, 0.2, 1), target)

    target = targets[1]
    assert np.allclose(alpha_eff(980, 0, 1), target)

    target = targets[2]
    assert np.allclose(alpha_eff(700, 0.2, 1), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e