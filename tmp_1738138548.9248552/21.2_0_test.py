from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def m_eff(x, m0=1):
    '''Calculates the effective mass of GaAlAs for a given aluminum mole fraction x using the minimum of the effective masses.
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
    
    # Calculate the relative effective mass using the minimum of the masses
    mr = min(m_e, m_lh, m_hh)
    
    return mr



# Background: The effective absorption coefficient, α, of a semiconductor material like Al_xGa_{1-x}As is influenced by the material's band structure and the interaction of photons with the electronic states. The absorption coefficient is related to the probability of photon absorption, which depends on the energy of the photons (related to their wavelength λ) and the electronic properties of the material, such as the effective mass of charge carriers. The effective mass affects the density of states and the transition probabilities. The absorption coefficient can be modeled as a function of the wavelength and the material composition, with a scaling factor C to account for other material-specific properties. The constants involved include the electron charge (e), the speed of light in vacuum (c), and the reduced Planck constant (ħ), which are fundamental in calculating the energy of photons and their interaction with the material.

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
    h_bar = 1.05e-34  # Reduced Planck constant in J·s

    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9

    # Calculate the photon energy E = hc/λ
    E_photon = (h_bar * c) / lambda_m

    # Calculate the effective mass using the given function
    m_r = m_eff(x)

    # Calculate the absorption coefficient using a model that includes the effective mass
    # This is a simplified model where the absorption coefficient is proportional to the photon energy
    # and inversely proportional to the effective mass, scaled by C.
    alpha_x = C * (E_photon / m_r)

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