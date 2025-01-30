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
    Jw = J0 * (gw / g0)
    
    return Jw



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
    
    # Use the gain function to calculate the gain coefficient at threshold
    gw = gain(nw, Gamma_w, alpha, L, R1, R2)
    
    # Use the current_density function to calculate the current density
    Jw = current_density(gw, g0, J0)
    
    # Calculate the threshold current using the injected current density
    Ith = Jw * w * L * nw / eta
    
    return Ith


try:
    targets = process_hdf5_to_tuple('42.3', 4)
    target = targets[0]
    assert (threshold_current(1, 0.02, 20, 0.0001, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4)>10**50) == target

    target = targets[1]
    assert np.allclose(threshold_current(10, 0.1, 20, 0.1, 0.3, 0.3, 3000, 200, 0.6, 2*10**-4), target)

    target = targets[2]
    assert np.allclose(threshold_current(1, 0.02, 20, 0.1, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4), target)

    target = targets[3]
    assert np.allclose(threshold_current(5, 0.02, 20, 0.1, 0.3, 0.3, 3000, 200, 0.8, 2*10**-4), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e