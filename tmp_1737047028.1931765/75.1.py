import numpy as np



# Background: The Moon and Koshino model describes the electronic hopping between atoms in graphene, 
# taking into account both in-plane and out-of-plane interactions. The hopping parameter, -t(R_i, R_j), 
# is calculated using two types of transfer integrals: V_{pp\pi} and V_{pp\sigma}. 
# V_{pp\pi} is associated with the in-plane π-bonding, while V_{pp\sigma} is related to the out-of-plane 
# σ-bonding. These integrals decay exponentially with distance, characterized by a decay constant b. 
# The parameter d is the total distance between two atoms, and dz is the component of this distance 
# perpendicular to the graphene plane. The formula combines these integrals to account for the 
# orientation of the atomic orbitals relative to the plane.


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
    # Calculate V_{pp\pi} using the given formula
    V_pp_pi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_{pp\sigma} using the given formula
    V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the hopping parameter -t(R_i, R_j)
    hopping = V_pp_pi * (1 - (dz / d)**2) + V_pp_sigma * (dz / d)**2
    
    return hopping


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('75.1', 3)
target = targets[0]

assert np.allclose(hopping_mk(d=4.64872812, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
target = targets[1]

assert np.allclose(hopping_mk(d=2.68394443, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
target = targets[2]

assert np.allclose(hopping_mk(d=7.99182454, dz=6.50066046, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)
