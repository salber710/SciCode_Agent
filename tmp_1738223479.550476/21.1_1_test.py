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
    
    # Calculate the harmonic mean of m_lh and m_hh for the DOS effective mass of holes
    m_h = 2 / ((1/m_lh) + (1/m_hh))
    
    # The relative effective mass m_r is calculated using a different combination of effective masses
    # Here we use a weighted sum and then raise to the power of 0.75
    m_r = pow(0.5 * m_e + 0.5 * m_h, 0.75)
    
    return m_r


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