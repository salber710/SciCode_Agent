import numpy as np

# Background: 
# The gain coefficient at threshold, G_th, is a critical parameter in laser physics. It represents the gain required to 
# overcome the intrinsic losses and achieve lasing. The threshold gain condition is given by the equation:
# G_th = (1/L) * ln(1/(R1 * R2)) + alpha
# where:
# - L is the cavity length.
# - R1 and R2 are the reflectivities of the two mirrors forming the laser cavity.
# - alpha is the intrinsic loss coefficient.
# The gain coefficient g_w for a single quantum well is related to the threshold gain by:
# g_w = G_th / (nw * Gamma_w)
# where:
# - nw is the number of quantum wells.
# - Gamma_w is the optical confinement factor, which represents the fraction of the optical mode confined in the quantum well.


def gain(nw, Gamma_w, alpha, L, R1, R2):
    '''Calculates the single peak gain coefficient g_w.
    Input:
    nw (float): Quantum well number.
    Gamma_w (float): Confinement factor of the waveguide.
    alpha (float): Internal loss coefficient in cm^-1.
    L (float): Cavity length in cm.
    R1 (float): The reflectivities of mirror 1.
    R2 (float): The reflectivities of mirror 2.
    Output:
    gw (float): Gain coefficient g_w.
    '''
    # Calculate the threshold gain G_th
    G_th = (1 / L) * np.log(1 / (R1 * R2)) + alpha
    
    # Calculate the single peak gain coefficient g_w
    gw = G_th / (nw * Gamma_w)
    
    return gw



# Background: 
# In laser physics, the relationship between the gain coefficient and the injected current density is often described 
# using empirical models. The current density J_w required to achieve a certain modal gain g_w can be related to an 
# empirical gain g_0 and an empirical current density J_0. This relationship is typically expressed in the form of an 
# exponential or linear equation, depending on the specific characteristics of the laser material and structure. 
# A common empirical relation is given by:
# J_w = J_0 * exp(g_w / g_0)
# where:
# - J_w is the actual current density.
# - J_0 is the empirical current density corresponding to the empirical gain g_0.
# - g_w is the modal gain.
# - g_0 is the empirical gain coefficient.


def current_density(gw, g0, J0):
    '''Calculates the current density J_w as a function of the gain coefficient g_w.
    Input:
    gw (float): Gain coefficient.
    g0 (float): Empirical gain coefficient.
    J0 (float): Empirical factor in the current density equation.
    Output:
    Jw (float): Current density J_w.
    '''
    # Calculate the current density J_w using the empirical relation
    Jw = J0 * np.exp(gw / g0)
    
    return Jw


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('42.2', 3)
target = targets[0]

assert np.allclose(current_density(1000, 3000, 200), target)
target = targets[1]

assert np.allclose(current_density(200, 3000, 200), target)
target = targets[2]

assert np.allclose(current_density(2000, 3000, 200), target)
