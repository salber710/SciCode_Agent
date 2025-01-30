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


def alpha_eff(lambda_i, x, C=1):


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

    # Define a model for absorption coefficient that considers a reciprocal dependency on the square root of the photon energy
    # and directly proportional to the cube of the effective mass, scaled by C.
    alpha_x = C * (m_r**3) / np.sqrt(E_photon)

    return alpha_x



# Background: The absorption coefficient of a semiconductor material like Al_xGa_{1-x}As can be influenced by its composition and the wavelength of incident light. 
# The absorption coefficient, α, is a measure of how much light is absorbed per unit distance in the material. 
# For Al_xGa_{1-x}As, the absorption coefficient can be normalized by the absorption coefficient of pure GaAs at a reference wavelength, λ_0. 
# This normalization allows us to compare the absorption properties of the alloy with those of the pure material. 
# The function alpha_eff(lambda_i, x, C=1) computes the effective absorption coefficient for Al_xGa_{1-x}As, 
# and we can use this to find the actual absorption coefficient by scaling it with the known absorption coefficient of GaAs at λ_0.


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
    
    # Calculate the effective absorption coefficient for Al_xGa_{1-x}As at the given wavelength
    alpha_x = alpha_eff(lambda_i, x, C=1)
    
    # Calculate the effective absorption coefficient for pure GaAs at the reference wavelength
    alpha_gaas = alpha_eff(lambda0, 0, C=1)
    
    # Normalize the absorption coefficient by the reference absorption coefficient
    alpha_final = alpha_x * (alpha0 / alpha_gaas)
    
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