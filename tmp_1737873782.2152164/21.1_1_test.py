from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def m_eff(x, m0=1):
    '''Calculates the relative effective mass of Al_xGa_{1-x}As for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in Al_xGa_{1-x}As.
    m0 (float): electron rest mass (can be reduced to 1 as default).
    Output:
    mr (float): Relative effective mass of Al_xGa_{1-x}As.
    '''
    
    # Calculate the effective masses based on the aluminum mole fraction x
    m_e = (0.0637 + 0.083 * x) * m0  # Effective electron mass
    m_lh = (0.087 + 0.063 * x) * m0  # Effective light hole mass
    m_hh = (0.50 + 0.29 * x) * m0    # Effective heavy hole mass

    # Calculate the density of states effective mass
    # This is typically calculated as a harmonic mean for semiconductors
    # However, for simplicity, let's take the geometric mean of the hole masses
    m_h_dos = (m_hh**(3/2) + m_lh**(3/2))**(2/3)

    # Relative effective mass for the density of states
    mr = (m_e * m_h_dos)**(1/2) / m0
    
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