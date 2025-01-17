import numpy as np

# Background: The density of states (DOS) effective mass is a concept used in semiconductor physics to describe the 
# effective mass of charge carriers (electrons and holes) in a semiconductor material. It is a measure of how the 
# energy levels are distributed in the material and is crucial for understanding the electronic properties of 
# semiconductors. For a compound semiconductor like Al_xGa_{1-x}As, the effective masses of electrons and holes 
# depend on the composition, specifically the aluminum mole fraction x. The effective mass for electrons (m_e), 
# light holes (m_lh), and heavy holes (m_hh) are given as functions of x. The relative effective mass m_r is 
# calculated using the formula: m_r = ((m_e * m_hh * m_lh)^(1/3)) / m0, where m0 is the electron rest mass.

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
    
    # Calculate the relative effective mass
    mr = ((m_e * m_lh * m_hh) ** (1/3)) / m0
    
    return mr


# Background: The effective absorption coefficient, α_x, of a semiconductor material like Al_xGa_{1-x}As is a measure of how much light is absorbed per unit distance as it travels through the material. This coefficient depends on the material's electronic properties, which are influenced by the composition (x) and the wavelength (λ) of the incident light. The absorption process is related to the transition of electrons between energy bands, which is influenced by the effective mass of the charge carriers. The effective mass affects the density of states and the joint density of states, which in turn influence the absorption coefficient. The absorption coefficient can be calculated using the formula: α_x = C * (m_r * e^2) / (h * c * λ), where m_r is the relative effective mass, e is the electron charge, h is the reduced Planck constant, c is the speed of light, and λ is the wavelength of the incident light.


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
    h_bar = 1.0545718e-34  # Reduced Planck constant in J·s

    # Convert wavelength from nm to m
    lambda_m = lambda_i * 1e-9

    # Calculate the relative effective mass using the given function
    m_r = m_eff(x, m0=1)

    # Calculate the effective absorption coefficient
    alpha_x = C * (m_r * e**2) / (h_bar * c * lambda_m)

    return alpha_x



# Background: The absorption coefficient of a semiconductor material like Al_xGa_{1-x}As can be influenced by the 
# composition of the material and the wavelength of the incident light. The absorption coefficient at a given 
# wavelength and composition can be normalized by the absorption coefficient at a reference wavelength for a 
# specific composition, such as pure GaAs (x=0). This normalization allows for comparison across different 
# compositions and wavelengths. The function alpha_eff(lambda, x, C=1) calculates the effective absorption 
# coefficient for a given composition and wavelength. To find the actual absorption coefficient, we can use 
# the ratio of the effective absorption coefficient at the desired wavelength and composition to the effective 
# absorption coefficient at the reference wavelength for pure GaAs, multiplied by the known absorption 
# coefficient at the reference wavelength.


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
    # Calculate the effective absorption coefficient at the given wavelength and composition
    alpha_x = alpha_eff(lambda_i, x, C=1)
    
    # Calculate the effective absorption coefficient at the reference wavelength for pure GaAs
    alpha_ref = alpha_eff(lambda0, 0, C=1)
    
    # Normalize the absorption coefficient by the reference absorption coefficient
    alpha_final = (alpha_x / alpha_ref) * alpha0
    
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
