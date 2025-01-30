from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: 
# The gain coefficient at threshold, G_th, is a critical parameter in laser physics. It represents the gain required to 
# overcome the intrinsic losses and achieve lasing. The threshold gain condition is given by the equation:
# G_th = (1/L) * ln(1/(R1 * R2)) + alpha
# where:
# - L is the cavity length.
# - R1 and R2 are the reflectivities of the two mirrors (facets) of the laser cavity.
# - alpha is the intrinsic loss coefficient.
# The gain coefficient g_w for a single quantum well is related to the threshold gain G_th by:
# g_w = G_th / (nw * Gamma_w)
# where:
# - nw is the number of quantum wells.
# - Gamma_w is the optical confinement factor, which describes how well the optical mode is confined to the active region.


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


try:
    targets = process_hdf5_to_tuple('42.1', 3)
    target = targets[0]
    assert np.allclose(gain(1, 0.02, 20, 0.1, 0.3, 0.3), target)

    target = targets[1]
    assert np.allclose(gain(1, 0.02, 20, 0.1, 0.8, 0.8), target)

    target = targets[2]
    assert np.allclose(gain(5, 0.02, 20, 0.1, 0.3, 0.3), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e