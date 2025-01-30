import numpy as np



# Background: The density of states (DOS) effective mass is a concept used in semiconductor physics to describe the 
# effective mass of charge carriers (electrons and holes) in a semiconductor material. It is a measure of how the 
# density of available electronic states changes with energy. For a compound semiconductor like Al_xGa_{1-x}As, 
# the effective masses of electrons and holes depend on the composition, specifically the aluminum mole fraction x. 
# The effective mass of electrons (m_e), light holes (m_lh), and heavy holes (m_hh) are given as functions of x. 
# The relative effective mass m_r is calculated using the formula:
# m_r = ((m_e * (m_hh^2/3) * (m_lh^1/3))^(1/3)) / m0
# where m0 is the electron rest mass. This formula accounts for the contributions of both heavy and light holes 
# to the density of states.


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
    
    # Calculate the relative effective mass m_r
    mr = ((m_e * (m_hh**(2/3)) * (m_lh**(1/3)))**(1/3)) / m0
    
    return mr

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.1', 3)
target = targets[0]

assert np.allclose(m_eff(0, 1), target)
target = targets[1]

assert np.allclose(m_eff(0.2, 1), target)
target = targets[2]

assert np.allclose(m_eff(0.6, 1), target)
