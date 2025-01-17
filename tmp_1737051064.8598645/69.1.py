import numpy as np



# Background: In a semi-infinite system of layered electron gas (LEG), the Coulomb interaction between two electrons
# is influenced by the dielectric constant of the material and the spatial separation between the layers. The interaction
# can be described in terms of a form factor, which is a function of the in-plane momentum transfer (q) and the positions
# of the electrons in the layers (z = l1*d and z' = l2*d). The form factor f(q;z,z') is derived by Fourier transforming
# the Coulomb potential with respect to the in-plane coordinates. The dielectric constant (bg_eps) modifies the effective
# interaction strength, and the layer spacing (d) determines the separation between the electron layers. The form factor
# is crucial for understanding the screening effects and the effective interaction in the layered system.


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
    # Calculate the distance between the layers in the z-direction
    z1 = l1 * d
    z2 = l2 * d
    delta_z = np.abs(z1 - z2)
    
    # Calculate the form factor using the exponential decay due to the layer separation
    form_factor = np.exp(-q * delta_z) / bg_eps
    
    return form_factor


from scicode.parse.parse import process_hdf5_to_tuple

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
