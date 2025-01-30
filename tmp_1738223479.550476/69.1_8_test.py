from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np





def f_V(q, d, bg_eps, l1, l2):
    '''Compute the form factor f(q;l1,l2) using a unique methodology
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
    
    # Utilize the modified Bessel function of the first kind for a unique effect
    bessel_i_term = iv(0.5, q * np.log1p(abs(z1 - z2) + d))

    # Introduce a polynomial modulation term based on the layer spacing
    polynomial_modulation = (1 + (z1 * z2) / (d**2 + q**2))

    # Apply a unique exponential damping factor involving a cubic root
    damping_factor = np.exp(-q**(1/3) * np.abs(z1 - z2) / (bg_eps * d))

    # Combine these components to form the final form factor
    form_factor = bessel_i_term * polynomial_modulation * damping_factor

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