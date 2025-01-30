from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The density of states (DOS) effective mass is a concept used in semiconductor physics to describe the
# effective mass of charge carriers (electrons and holes) in a semiconductor material. It is a measure of how the
# energy levels are distributed in the material and is crucial for understanding the electronic properties of
# semiconductors. For a compound semiconductor like Al_xGa_{1-x}As, the effective masses of electrons and holes
# depend on the composition, specifically the aluminum mole fraction x. The effective mass of electrons (m_e),
# light holes (m_lh), and heavy holes (m_hh) are given as functions of x. The relative effective mass m_r is
# calculated using the geometric mean of these masses, which is a common approach to approximate the DOS effective
# mass in semiconductors.

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
    
    # Calculate the relative effective mass using the geometric mean
    mr = (m_e * m_lh * m_hh) ** (1/3)
    
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