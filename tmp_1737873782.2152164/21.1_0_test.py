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


try:
    targets = process_hdf5_to_tuple('21.1', 3)
    target = targets[0]
    assert np.allclose(m_eff(0, 1), target)

    target = targets[1]
    assert np.allclose(m_eff(0.2, 1), target)

    target = targets[2]
    assert np.allclose(m_eff(0.6, 1), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e