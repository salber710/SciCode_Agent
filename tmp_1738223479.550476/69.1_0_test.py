from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: In a semi-infinite system of layered electron gas (LEG), the Coulomb interaction between electrons can be complex due to the presence of layers and dielectric properties. When two electrons are positioned at different layers, the interaction can be described using a form factor. This form factor, f(q;z,z'), is obtained by Fourier transforming the Coulomb potential with respect to the in-plane vector difference, resulting in a dependence on the in-plane momentum q and the z-coordinates of the electron layers. For a system with dielectric constant ε, the form factor at layers l1 and l2, with spacing d, incorporates these effects, and is crucial for understanding electron interactions within the LEG. The function f_V calculates this form factor for given parameters.

def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1, l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    # Calculate the distance between the layers in the z-direction
    z1 = l1 * d
    z2 = l2 * d
    delta_z = abs(z1 - z2)

    # Calculate the exponential decay factor due to the dielectric and distance
    exponential_decay = np.exp(-q * delta_z)

    # Form factor considering the dielectric constant and spacing
    form_factor = (2 * np.pi / bg_eps) * exponential_decay

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