import numpy as np

# Background: 
# The gain coefficient at threshold, G_th, is a critical parameter in laser physics, particularly for semiconductor lasers.
# It represents the gain required to overcome the losses in the laser cavity and achieve lasing. The threshold gain is 
# determined by the balance between the gain provided by the active medium and the losses due to internal absorption 
# and mirror losses. The gain coefficient g_w can be calculated using the following formula:
# 
# G_th = (1 / (2 * L)) * ln(1 / (R1 * R2)) + alpha
# 
# where:
# - L is the cavity length in cm.
# - R1 and R2 are the reflectivities of the two mirrors (facets) of the laser cavity.
# - alpha is the intrinsic loss coefficient in cm^-1.
# 
# The gain coefficient g_w is then related to the threshold gain G_th by considering the number of quantum wells (n_w)
# and the optical confinement factor (Gamma_w) as follows:
# 
# g_w = G_th / (n_w * Gamma_w)
# 
# This formula accounts for the distribution of the gain across multiple quantum wells and the confinement of the optical
# mode within the active region.


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
    if L <= 0:
        raise ValueError("Cavity length must be positive.")
    if nw <= 0:
        raise ValueError("Quantum well number must be positive.")
    if Gamma_w <= 0:
        raise ValueError("Confinement factor must be positive.")
    if R1 <= 0 or R2 <= 0:
        raise ValueError("Reflectivities must be positive and non-zero.")
    if alpha < 0:
        raise ValueError("Internal loss coefficient alpha must be non-negative.")
    
    if R1 == 1 and R2 == 1:
        G_th = alpha  # Special case where no gain is needed from mirrors
    else:
        G_th = (1 / (2 * L)) * np.log(1 / (R1 * R2)) + alpha
    
    gw = G_th / (nw * Gamma_w)
    
    return gw


# Background: In semiconductor laser physics, the relationship between the gain coefficient and the injected current
# density is often described by an empirical relation. This relation helps in understanding how the gain in the laser
# medium is influenced by the current injected into the device. The empirical relation typically involves a reference
# gain coefficient (g_0) and a corresponding reference current density (J_0). The actual current density (J_w) required
# to achieve a certain gain (g_w) can be calculated using these empirical parameters. The relationship is often linear
# or follows a simple proportionality, allowing for straightforward calculation of the current density from the gain.

def current_density(gw, g0, J0):
    '''Calculates the current density J_w as a function of the gain coefficient g_w.
    Input:
    gw (float): Gain coefficient.
    g0 (float): Empirical gain coefficient.
    J0 (float): Empirical factor in the current density equation.
    Output:
    Jw (float): Current density J_w.
    '''
    if g0 <= 0:
        raise ValueError("Empirical gain coefficient g0 must be positive.")
    if J0 < 0:
        raise ValueError("Empirical current density J0 must be non-negative.")
    
    # Calculate the current density using the empirical relation
    Jw = J0 * (gw / g0)
    
    return Jw



# Background: In semiconductor laser physics, the threshold current (I_th) is the minimum current required to achieve
# lasing. It is determined by the balance between the gain provided by the active medium and the losses in the laser
# cavity. The threshold current can be calculated using the gain coefficient and the injected current density. The
# threshold current is related to the injected current density (J_w) and the physical dimensions of the laser device,
# such as the cavity length (L) and the device width (w). The injection quantum efficiency (eta) also plays a role in
# determining the threshold current, as it represents the efficiency with which the injected carriers contribute to
# the gain. The threshold current can be calculated using the formula:
# 
# I_th = J_w * L * w * n_w / eta
# 
# where:
# - J_w is the injected current density.
# - L is the cavity length.
# - w is the device width.
# - n_w is the number of quantum wells.
# - eta is the injection quantum efficiency.


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
    # Calculate the gain coefficient g_w using the gain function
    gw = gain(nw, Gamma_w, alpha, L, R1, R2)
    
    # Calculate the injected current density J_w using the current_density function
    Jw = current_density(gw, g0, J0)
    
    # Calculate the threshold current I_th
    Ith = Jw * L * w * nw / eta
    
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
