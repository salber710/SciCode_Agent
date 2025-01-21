import numpy as np



# Background: The density of states (DOS) effective mass, also known as the relative effective mass (m_r), is a 
# measure used in solid-state physics to simplify the calculation of the number of electronic states at a given 
# energy. In semiconductor materials like Al_xGa_{1-x}As, the effective mass of charge carriers (electrons and 
# holes) is not constant but varies with the composition of the material. For Al_xGa_{1-x}As, the effective masses 
# of electrons (m_e), light holes (m_lh), and heavy holes (m_hh) can be expressed as linear functions of the 
# aluminum mole fraction x. These effective masses are used to calculate the DOS effective mass for both electrons 
# and holes, which is crucial for understanding the material's electronic properties. The DOS effective mass for 
# electrons (m_re) is the same as m_e, and for holes (m_rh), it's a geometrical mean of m_lh and m_hh. The overall 
# DOS effective mass (m_r) is a weighted average of these masses.


def m_eff(x, m0):
    '''Calculates the effective mass of GaAlAs for a given aluminum mole fraction x.
    Input:
    x (float): Aluminum mole fraction in GaAlAs.
    m0 (float): electron rest mass (can be reduced to 1 as default).
    Output:
    mr (float): Effective mass of GaAlAs.
    '''
    # Calculate effective electron mass
    m_e = (0.0637 + 0.083 * x) * m0
    
    # Calculate effective light hole mass
    m_lh = (0.087 + 0.063 * x) * m0
    
    # Calculate effective heavy hole mass
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Density of states effective mass for holes is the geometric mean of m_hh and m_lh
    m_rh = np.sqrt(m_hh * m_lh)
    
    # The DOS effective mass m_r is calculated as a simple average for this context
    mr = (m_e + m_rh) / 2
    
    return mr

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.1', 3)
target = targets[0]

assert np.allclose(m_eff(0, 1), target)
target = targets[1]

assert np.allclose(m_eff(0.2, 1), target)
target = targets[2]

assert np.allclose(m_eff(0.6, 1), target)
