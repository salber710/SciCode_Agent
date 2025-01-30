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
    h = 1.0545718e-34  # Reduced Planck constant in J·s
    c = 3e8            # Speed of light in m/s

    # Convert wavelength from nm to meters
    lambda_m = lambda_i * 1e-9
    
    # Calculate photon energy (E = h*c/λ)
    E_photon = h * c / lambda_m
    
    # Use the effective mass function m_eff to get the effective mass
    m_r = m_eff(x, m0=1)
    
    # Implement a novel approach: α ~ C * exp(-(E_photon / m_r)^2) * log(E_photon + 1)
    # This introduces an exponential decay of the squared energy-to-mass ratio and a logarithmic dependency
    alpha_x = C * (2.71828**(-(E_photon / m_r)**2)) * (E_photon + 1)**0.5
    
    return alpha_x



def alpha(lambda_i, x, lambda0, alpha0):
    '''Computes the absorption coefficient for a given wavelength and Al composition,
    normalized by the absorption coefficient at a reference wavelength for pure GaAs.
    Input:
    lambda_i (float): Wavelength of the incident light (nm).
    x (float): Aluminum composition in the AlxGa1-xAs alloy.
    lambda0 (float): Reference wavelength (nm) for pure GaAs (x=0).
    alpha0 (float): Absorption coefficient at the reference wavelength for pure GaAs.
    Output:
    alpha_final (float): Normalized absorption coefficient in m^-1.
    '''


    # New model using a combination of cosine modulation and reciprocal adjustments
    a = 0.4  # Arbitrary constant for cosine modulation on x
    b = 0.15 # Arbitrary constant for inverse dependence on wavelength difference

    # Compute the modified absorption coefficient
    delta_lambda = lambda_i - lambda0
    cosine_term = (1 + a * np.cos(x * np.pi))
    reciprocal_term = (1 + b / (1 + (delta_lambda / lambda0) ** 2))

    alpha_x = alpha0 * cosine_term * reciprocal_term

    # Ensure that the absorption coefficient remains non-negative
    alpha_final = max(alpha_x, 0)

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