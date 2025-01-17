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



# Background: 
# The threshold current (I_th) of a laser is the minimum current required to achieve lasing. It is determined by the 
# balance between the gain provided by the active medium and the losses in the laser cavity. The threshold current can 
# be calculated using the injected current density (J_w) and the physical dimensions of the laser. The injected current 
# density is the current per unit area required to achieve the necessary gain for lasing. The threshold current is 
# given by the equation:
# I_th = J_w * A / eta
# where:
# - I_th is the threshold current.
# - J_w is the injected current density.
# - A is the cross-sectional area of the laser, which can be calculated as the product of the cavity length (L) and 
#   the device width (w).
# - eta is the injection quantum efficiency, which accounts for the fraction of the injected carriers that contribute 
#   to the lasing process.


def threshold_current(nw, Gamma_w, alpha, L, R1, R2, g0, J0, eta, w):
    '''Calculates the threshold current.
    Input:
    nw (float): Quantum well number.
    Gamma_w (float): Confinement factor of the waveguide.
    alpha (float): Internal loss coefficient.
    L (float): Cavity length.
    R1 (float): The reflectivities of mirror 1.
    R2 (float): The reflectivities of mirror 2.
    g0 (float): Empirical gain coefficient.
    J0 (float): Empirical factor in the current density equation.
    eta (float): injection quantum efficiency.
    w (float): device width.
    Output:
    Ith (float): threshold current Ith
    '''
    # Calculate the single peak gain coefficient g_w using the gain function
    gw = gain(nw, Gamma_w, alpha, L, R1, R2)
    
    # Calculate the injected current density J_w using the current_density function
    Jw = current_density(gw, g0, J0)
    
    # Calculate the cross-sectional area A of the laser
    A = L * w
    
    # Calculate the threshold current I_th
    Ith = Jw * A / eta
    
    return Ith


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('42.3', 4)
target = targets[0]

assert (threshold_current(1, 0.02, 20, 0.0001, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4)>10**50) == target
target = targets[1]

assert np.allclose(threshold_current(10, 0.1, 20, 0.1, 0.3, 0.3, 3000, 200, 0.6, 2*10**-4), target)
target = targets[2]

assert np.allclose(threshold_current(1, 0.02, 20, 0.1, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4), target)
target = targets[3]

assert np.allclose(threshold_current(5, 0.02, 20, 0.1, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4), target)
