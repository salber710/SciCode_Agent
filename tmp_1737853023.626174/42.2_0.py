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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('42.2', 3)
target = targets[0]

assert np.allclose(current_density(1000, 3000, 200), target)
target = targets[1]

assert np.allclose(current_density(200, 3000, 200), target)
target = targets[2]

assert np.allclose(current_density(2000, 3000, 200), target)
