from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The density of states (DOS) effective mass, or relative effective mass, is used in semiconductor physics
# to characterize the number of available electron or hole states at a given energy level. For compound semiconductors
# like Al_xGa_{1-x}As, the effective masses of the charge carriers depend on the composition of the material, which is
# expressed by the aluminum mole fraction, x. The effective electron mass (m_e), heavy hole mass (m_hh), and light hole
# mass (m_lh) for Al_xGa_{1-x}As are given as functions of x and the electron rest mass m_0. The DOS effective mass for
# holes can be approximated by the geometric mean of the heavy and light hole masses. The relative effective mass m_r is
# a dimensionless parameter representing the DOS effective mass relative to the rest mass of an electron.

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
    
    # Calculate the DOS effective mass for holes using the geometric mean of m_lh and m_hh
    m_h = np.sqrt(m_lh * m_hh)
    
    # The relative effective mass m_r is a combination of electron and hole effective masses
    # For this DOS effective mass calculation, the relative effective mass is often approximated as:
    # m_r = (m_e * m_h)^(2/3)
    m_r = (m_e * m_h)**(2/3)
    
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