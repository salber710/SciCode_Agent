import numpy as np



# Background: The Moon and Koshino hopping parameterization is used to model the electron hopping between atoms in a bilayer graphene system. 
# This involves calculating two types of transfer integrals: V_{pp\pi} and V_{pp\sigma}, which describe the π-bonding and σ-bonding between 
# carbon atoms, respectively. The hopping parameter, -t(R_i, R_j), is a combination of these two transfer integrals and depends on the 
# geometric configuration of the atoms, specifically the distance d between them and the out-of-plane distance dz. 
# The expressions for V_{pp\pi} and V_{pp\sigma} involve exponential decay based on their respective decay lengths, and they are affected by 
# the distances relative to the monolayer graphene's nearest-neighbor distance a0 and the interlayer distance d0.


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
    # Calculate the ratio of out-of-plane distance to total distance
    dz_over_d = dz / d
    
    # Calculate V_{pp\pi} using the given exponential decay formula
    V_pp_pi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_{pp\sigma} using the given exponential decay formula
    V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the hopping parameter -t(R_i, R_j)
    hopping = V_pp_pi * (1 - dz_over_d**2) + V_pp_sigma * (dz_over_d**2)
    
    return hopping

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('75.1', 3)
target = targets[0]

assert np.allclose(hopping_mk(d=4.64872812, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
target = targets[1]

assert np.allclose(hopping_mk(d=2.68394443, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
target = targets[2]

assert np.allclose(hopping_mk(d=7.99182454, dz=6.50066046, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
