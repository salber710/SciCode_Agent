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

    # Calculate the mirror loss using the reflectivity of the facets
    mirror_loss = (1 / (2 * L)) * np.log(1 / (R1 * R2))

    # Calculate the gain coefficient at threshold
    gw = (alpha + mirror_loss) / (nw * Gamma_w)

    return gw



def current_density(gw, g0, J0):
    '''Calculates the current density J_w as a function of the gain coefficient g_w.
    Input:
    gw (float): Gain coefficient.
    g0 (float): Empirical gain coefficient.
    J0 (float): Empirical factor in the current density equation.
    Output:
    Jw (float): Current density J_w.
    '''

    # Calculate the current density using the empirical relation
    Jw = J0 * np.exp(gw / g0)

    return Jw


try:
    targets = process_hdf5_to_tuple('42.2', 3)
    target = targets[0]
    assert np.allclose(current_density(1000, 3000, 200), target)

    target = targets[1]
    assert np.allclose(current_density(200, 3000, 200), target)

    target = targets[2]
    assert np.allclose(current_density(2000, 3000, 200), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e