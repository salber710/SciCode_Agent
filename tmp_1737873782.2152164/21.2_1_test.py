from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def m_eff(x, m0=1):
    '''Calculates the effective density of states mass (relative effective mass) of Al_xGa_{1-x}As for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in Al_xGa_{1-x}As.
    m0 (float): Electron rest mass, default is set to 1 for relative calculations.
    Output:
    mr (float): Effective density of states mass (unitless).
    '''
    # Calculate effective electron mass
    m_e = (0.0637 + 0.083 * x) * m0
    
    # Calculate effective light hole mass
    m_lh = (0.087 + 0.063 * x) * m0
    
    # Calculate effective heavy hole mass
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Calculate relative effective mass for density of states
    mr = ((m_e * (m_hh ** (2/3)) * (m_lh ** (1/3))) ** (1/3)) / m0
    
    return mr



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
    e = 1.6e-19  # Elementary charge in C
    c = 3e8      # Speed of light in m/s
    h_bar = 1.05e-34  # Reduced Planck's constant in J.s

    # Convert wavelength from nm to m
    lambda_m = lambda_i * 1e-9

    # Calculate the photon energy in Joules
    E_photon = (h_bar * c) / lambda_m

    # Calculate the effective density of states mass using the given m_eff function
    mr = m_eff(x)

    # Calculate the effective absorption coefficient alpha_x
    # Assuming a proportional relationship between the absorption and photon energy/mass
    alpha_x = C * e * E_photon / (mr * c)

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