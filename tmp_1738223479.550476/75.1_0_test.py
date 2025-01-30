from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: 
# The Moon and Koshino model for hopping parameters in graphene involves calculating the transfer
# integrals between atomic orbitals. These integrals are denoted as V_{pp\pi} and V_{pp\sigma}.
# The V_{pp\pi} describes the hopping interaction between in-plane orbitals, and V_{pp\sigma} 
# describes the hopping interaction between orbitals that are out-of-plane. 
# The expressions for these transfer integrals incorporate exponential decay terms, which depend 
# on the distance between atoms, d, and specific parameters a0 and d0 that relate to the structure 
# of graphene. The decay factor b influences how quickly the hopping terms decrease with distance.
# The hopping energy -t is a combination of these two integrals, weighted by the geometric factor 
# (d_z/d)^2, which accounts for the component of distance perpendicular to the graphene plane.


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
    # Calculate V_{pp\pi} using the exponential decay factor and given parameters
    V_pp_pi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_{pp\sigma} using the exponential decay factor and given parameters
    V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the geometric factor (d_z/d)^2
    dz_over_d_squared = (dz / d) ** 2
    
    # Compute the hopping parameter -t
    hopping = V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared
    
    return hopping


try:
    targets = process_hdf5_to_tuple('75.1', 3)
    target = targets[0]
    assert np.allclose(hopping_mk(d=4.64872812, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)

    target = targets[1]
    assert np.allclose(hopping_mk(d=2.68394443, dz=0.0, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)

    target = targets[2]
    assert np.allclose(hopping_mk(d=7.99182454, dz=6.50066046, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e