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
    # Calculate the effective electron mass (m_e)
    m_e = (0.0637 + 0.083 * x) * m0
    
    # Calculate the effective light hole mass (m_lh)
    m_lh = (0.087 + 0.063 * x) * m0
    
    # Calculate the effective heavy hole mass (m_hh)
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Calculate the density of states effective mass (m_r)
    # Using the formula for the effective mass of the density of states in a semiconductor
    mr = ((m_e * m_hh * m_lh) ** (1/3)) / m0

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