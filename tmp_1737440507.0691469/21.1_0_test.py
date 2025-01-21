import numpy as np



# Background: 
# The density of states (DOS) effective mass is a concept in semiconductor physics 
# that describes how the energy levels are distributed in a semiconductor material.
# In a semiconductor like Al_xGa_{1-x}As, which is a mixture of Aluminum (Al) and Gallium Arsenide (GaAs),
# the effective mass of charge carriers (electrons and holes) can vary with the composition (x) of the material.
# The DOS effective mass is important for determining the electronic properties of the material, 
# such as carrier concentration and mobility.
# The relative effective mass (m_r) is given by the geometric mean of the effective masses of the 
# conduction band (electron) and the valence band (heavy and light holes).
# For the conduction band, the effective electron mass is given by m_e = (0.0637 + 0.083 * x) * m_0.
# In the valence band, the heavy hole and light hole masses are given by:
# m_hh = (0.50 + 0.29 * x) * m_0 and m_lh = (0.087 + 0.063 * x) * m_0.
# The effective mass for density of states (m_r) is often calculated for semiconductors as:
# m_r = ((m_hh^2 * m_lh)^(1/3)) / m0, where m0 is the electron rest mass.

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
    
    # Calculate heavy hole mass
    m_hh = (0.50 + 0.29 * x) * m0
    
    # Calculate light hole mass
    m_lh = (0.087 + 0.063 * x) * m0
    
    # Calculate the density of states effective mass (DOS effective mass)
    # Using the geometric mean for the valence band (heavy and light holes)
    m_v = (m_hh**2 * m_lh)**(1/3)
    
    # Calculate the relative effective mass m_r
    mr = m_v / m0  # Normalizing by the electron rest mass
    
    return mr

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('21.1', 3)
target = targets[0]

assert np.allclose(m_eff(0, 1), target)
target = targets[1]

assert np.allclose(m_eff(0.2, 1), target)
target = targets[2]

assert np.allclose(m_eff(0.6, 1), target)
