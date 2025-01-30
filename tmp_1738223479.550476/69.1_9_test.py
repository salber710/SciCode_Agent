from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




def f_V(q, d, bg_eps, l1, l2):
    '''Compute the form factor f(q;l1,l2) using a completely different approach
    Input
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1, l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    # Compute the z-coordinates for the given layers
    z1 = l1 * d
    z2 = l2 * d
    
    # Introduce a sigmoid function to smoothly interpolate the interaction
    sigmoid_factor = 1 / (1 + np.exp(-q * (z1 - z2) / d))
    
    # Use a Lambert W function-based term for a unique oscillatory behavior
    lambert_w_term = np.real(np.exp(np.log(q * d + 1) - np.log(bg_eps) + np.log(np.abs(z1 - z2) + 1)))
    
    # Implement a quadratic damping factor with a distinct scaling
    quadratic_damping = np.exp(-q**2 * (z1**2 + z2**2) / (bg_eps**2 * d**2))
    
    # Combine these terms to form the final form factor
    form_factor = sigmoid_factor * lambert_w_term * quadratic_damping

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