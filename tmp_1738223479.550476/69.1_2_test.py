from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




def f_V(q, d, bg_eps, l1, l2):
    '''Calculate the form factor f(q;l1,l2) using a distinct approach
    Input
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1, l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    # Position of layers in the z-direction
    z1 = l1 * d
    z2 = l2 * d
    
    # Instead of using absolute difference, calculate a weighted difference
    weighted_z_diff = (z1 - z2) * (1 + 0.5 * np.tanh(q * d))

    # Use a hyperbolic cosine term for decay to introduce a different functional form
    hyperbolic_term = np.cosh(q * weighted_z_diff / bg_eps)

    # Incorporate a polynomial factor for further variation
    polynomial_factor = (1 + (q * d)**2 / bg_eps)

    # Final form factor combining all elements
    form_factor = (1 / polynomial_factor) * (1 / hyperbolic_term)

    return form_factor


try:
    targets = process_hdf5_to_tuple('69.1', 3)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    k_F = np.sqrt(2*np.pi*n_eff)   ###unit: A-1
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1 * k_F
    l1 = 1
    l2 = 3
    assert np.allclose(f_V(q,d,bg_eps,l1,l2), target)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    k_F = np.sqrt(2*np.pi*n_eff)   ###unit: A-1
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.01 * k_F
    l1 = 2
    l2 = 4
    assert np.allclose(f_V(q,d,bg_eps,l1,l2), target)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    k_F = np.sqrt(2*np.pi*n_eff)   ###unit: A-1
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.05 * k_F
    l1 = 0
    l2 = 2
    assert np.allclose(f_V(q,d,bg_eps,l1,l2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e