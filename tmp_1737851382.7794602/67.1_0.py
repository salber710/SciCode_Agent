import numpy as np



# Background: In a layered electron gas (LEG) system, the Coulomb interaction between two electrons
# is influenced by the dielectric constant of the material and the spatial separation between the layers.
# The interaction potential in a 3D system is given by V(r) = e^2 / (4 * pi * epsilon_0 * epsilon * r),
# where epsilon is the dielectric constant. In a 2D system, the Fourier transform of the Coulomb potential
# is used to simplify calculations, resulting in V_q = e^2 / (2 * epsilon * q) for in-plane momentum q.
# The form factor f(q;z,z') accounts for the layered structure and the positions of the electrons in the layers.
# It modifies the interaction potential to reflect the discrete nature of the layers and their separation.


def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1,l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''
    # Calculate the z positions of the two electrons
    z1 = l1 * d
    z2 = l2 * d
    
    # Calculate the form factor using the exponential decay due to the separation in the z-direction
    form_factor = np.exp(-q * np.abs(z1 - z2) / bg_eps)
    
    return form_factor

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('67.1', 3)
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
