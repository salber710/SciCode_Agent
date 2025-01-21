import numpy as np



# Background: The Moon and Koshino hopping parameterization is used to describe the electronic interactions
# in bilayer graphene. The hopping parameter `-t(R_i, R_j)` between two atoms at positions `R_i` and `R_j`
# depends on the distance vector `d = R_i - R_j`. The hopping is determined by two types of interactions:
# `V_{pp\pi}` and `V_{pp\sigma}`. `V_{pp\pi}` describes the in-plane π bonding, which decreases exponentially
# with distance and is dominant when the bond is primarily in-plane. `V_{pp\sigma}` describes the σ bonding,
# which becomes significant when the bond direction is out of the graphene plane. The formula for hopping involves
# these two components, weighted by the square of the cosine of the angle between the bond direction and the 
# perpendicular to the plane, represented by `dz/d`. The parameters `v_p0` and `v_s0` are the prefactors for 
# `V_{pp\pi}` and `V_{pp\sigma}` respectively, while `b` is related to the decay length of these interactions.

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


    # Calculate the angle factor (dz/d)^2
    dz_over_d_squared = (dz / d) ** 2
    
    # Calculate V_{pp\pi} using the given formula
    V_pp_pi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_{pp\sigma} using the given formula
    V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
    
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
