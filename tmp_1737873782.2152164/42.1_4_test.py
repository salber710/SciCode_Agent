from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




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

    # Calculate the mirror loss using the reflectivities of the two mirrors
    mirror_loss = (1 / (2 * L)) * np.log(1 / (R1 * R2))

    # Gain coefficient at threshold condition
    gw = (alpha + mirror_loss) / (Gamma_w * nw)

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