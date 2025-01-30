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
    # Calculate the threshold gain G_th
    G_th = (1 / (2 * L)) * np.log(1 / (R1 * R2)) + alpha
    
    # Calculate the gain coefficient g_w
    gw = G_th / (nw * Gamma_w)
    
    return gw

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('42.1', 3)
target = targets[0]

assert np.allclose(gain(1, 0.02, 20, 0.1, 0.3, 0.3), target)
target = targets[1]

assert np.allclose(gain(1, 0.02, 20, 0.1, 0.8, 0.8), target)
target = targets[2]

assert np.allclose(gain(5, 0.02, 20, 0.1, 0.3, 0.3), target)
