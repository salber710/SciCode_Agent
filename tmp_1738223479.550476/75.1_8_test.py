from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



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
    # Use a different mathematical series to approximate exponential
    def taylor_exp(x, terms=5):
        total = 1
        power = x
        factorial = 1
        for i in range(1, terms):
            total += power / factorial
            power *= x
            factorial *= (i + 1)
        return total
    
    # Calculate decay factors using the Taylor series approximation
    decay_pi = taylor_exp(-b * (d - a0))
    decay_sigma = taylor_exp(-b * (d - d0))
    
    # Calculate the transfer integrals
    V_pp_pi = v_p0 * decay_pi
    V_pp_sigma = v_s0 * decay_sigma
    
    # Compute the geometric factor (dz/d)^2
    dz_over_d_squared = (dz / d) ** 2
    
    # Compute the hopping parameter -t using a different logical structure
    hopping = (V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared)
    
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