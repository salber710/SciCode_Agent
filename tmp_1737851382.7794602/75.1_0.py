import numpy as np



# Background: The Moon and Koshino model describes the electronic hopping between atoms in a graphene system.
# The hopping parameter, -t(R_i, R_j), is calculated based on the distance vector d = R_i - R_j between two atoms.
# The model considers two types of interactions: pi (π) and sigma (σ) bonds. The pi bond interaction, V_ppπ, is
# dominant when the atoms are in-plane, while the sigma bond interaction, V_ppσ, becomes significant when there
# is an out-of-plane component (d_z) to the distance. The hopping parameter is a combination of these two
# interactions, weighted by the square of the ratio of the out-of-plane distance to the total distance.
# The exponential terms account for the decay of these interactions with distance, using parameters v_p0, v_s0,
# b, a0, and d0, which are specific to the Moon and Koshino model.


def hopping_mk(d, dz, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33):
    '''Parameterization from Moon and Koshino, Phys. Rev. B 85, 195458 (2012).
    Args:
        d: distance between two atoms (unit b,a.u.), float
        dz: out-of-plane distance between two atoms (unit b,a.u.), float
        v_p0: transfer integral between the nearest-neighbor atoms of monolayer graphene, MK parameter, float,unit eV
        v_s0: interlayer transfer integral between vertically located atoms, MK parameter, float,unit eV
        b: 1/b is the decay length of the transfer integral, MK parameter, float, unit (b,a.u.)^-1
        a0: nearest-neighbor atom distance of the monolayer graphene, MK parameter, float, unit (b,a.u.)
        d0: interlayer distance, MK parameter, float, (b,a.u.)
    Return:
        hopping: -t, float, eV
    '''
    # Calculate V_ppπ using the exponential decay formula
    V_pp_pi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_ppσ using the exponential decay formula
    V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the square of the ratio of the out-of-plane distance to the total distance
    dz_over_d_squared = (dz / d) ** 2
    
    # Calculate the hopping parameter -t(R_i, R_j)
    hopping = V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared
    
    return hopping

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('75.1', 3)
target = targets[0]

assert np.allclose(hopping_mk(d=4.64872812, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
target = targets[1]

assert np.allclose(hopping_mk(d=2.68394443, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
target = targets[2]

assert np.allclose(hopping_mk(d=7.99182454, dz=6.50066046, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
