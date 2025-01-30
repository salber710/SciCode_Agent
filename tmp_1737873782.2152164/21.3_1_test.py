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
    e = 1.6e-19  # Electron charge in C
    c = 3e8  # Speed of light in m/s
    hbar = 1.05e-34  # Reduced Planck constant in J·s
    
    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9
    
    # Calculate the photon energy
    E_photon = hbar * c / lambda_m  # Energy in Joules
    
    # Get the relative effective mass using the provided function
    m_r = m_eff(x, m0=1)  # m0=1 for relative calculation
    
    # Calculate the absorption coefficient using a hypothetical relation
    # This is a placeholder expression as the exact relationship was not given
    alpha_x = C * E_photon * m_r / (e * lambda_m)
    
    return alpha_x



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
    # Compute the effective absorption coefficient using the previous function
    C = alpha0  # Set C to alpha0 to match reference condition at lambda0
    alpha_x = alpha_eff(lambda_i, x, C)
    
    # Calculate the absorption coefficient ratio
    # alpha_relative = (alpha_x / alpha_eff(lambda0, 0, C)) * alpha0
    alpha_reference = alpha_eff(lambda0, 0, C)
    
    # Calculate the final absorption coefficient
    alpha_final = (alpha_x / alpha_reference) * alpha0
    
    return alpha_final


try:
    targets = process_hdf5_to_tuple('21.3', 4)
    target = targets[0]
    assert (alpha(850, 0.2, 850, 9000) == 0) == target

    target = targets[1]
    assert np.allclose(alpha(800, 0.1, 850, 8000), target)

    target = targets[2]
    assert np.allclose(alpha(700, 0.2, 850, 9000), target)

    target = targets[3]
    assert np.allclose(alpha(700, 0.1, 850, 9000), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e