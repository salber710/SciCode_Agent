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
    h_bar = 1.0545718e-34  # Reduced Planck constant in J·s
    c = 3e8                # Speed of light in m/s

    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9
    
    # Calculate photon energy using E = h*c/λ (E = (h-bar*2π)*c/λ)
    E_photon = (h_bar * 2 * 3.14159 * c) / lambda_m
    
    # Obtain the effective mass for the given aluminum composition
    m_r = m_eff(x, m0=1)
    
    # Implement a distinct formula: α ~ C * (E_photon^3) / (m_r^2)
    # This introduces a cubic dependency on the photon energy and a squared effective mass dependency
    alpha_x = C * (E_photon**3) / (m_r**2)
    
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