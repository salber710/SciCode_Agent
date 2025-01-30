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
    # Use a different mathematical approach to approximate exponential
    def exp_approx(x):
        # Using a more complex polynomial approximation for exponential
        return 1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24 + x**5 / 120

    # Calculate decay factors using the new approximation
    decay_pi = exp_approx(-b * (d - a0))
    decay_sigma = exp_approx(-b * (d - d0))

    # Calculate the transfer integrals V_{pp\pi} and V_{pp\sigma}
    V_pp_pi = v_p0 * decay_pi
    V_pp_sigma = v_s0 * decay_sigma

    # Calculate the squared ratio of dz to d using a nested operation
    dz_over_d_squared = (dz / d) ** 2

    # Compute the hopping parameter -t using a variation in arithmetic sequence
    hopping = V_pp_pi - V_pp_pi * dz_over_d_squared + V_pp_sigma * dz_over_d_squared

    return hopping




def mk(latvecs, basis, di, dj, ai, aj):
    '''Evaluate the Moon and Koshino hopping parameters Phys. Rev. B 85, 195458 (2012).
    Args:
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr; natoms: number of atoms within a unit cell
        di, dj (np.array): list of displacement indices for the hopping
        ai, aj (np.array): list of atomic basis indices for the hopping
    Return:
        hopping (np.array): a list with the same length as di
    '''

    def compute_hopping(d, dz):
        '''Custom Moon and Koshino hopping parameterization.
        Args:
            d: distance between two atoms (unit b,a.u.), float
            dz: out-of-plane distance between two atoms (unit b,a.u.), float
        Return:
            hopping: float, eV
        '''
        # Moon and Koshino parameters
        v_p0 = -2.7
        v_s0 = 0.48
        b = 1.17
        a0 = 2.68
        d0 = 6.33

        # Custom decay function using a trigonometric approach
        decay_pi = np.cos(b * (d - a0))
        decay_sigma = np.sin(b * (d - d0))

        V_pp_pi = v_p0 * decay_pi
        V_pp_sigma = v_s0 * decay_sigma

        dz_over_d_squared = (dz / d) ** 2

        hopping = V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared

        return hopping

    num_hoppings = len(di)
    hoppings = np.empty(num_hoppings)
    
    for idx in range(num_hoppings):
        # Calculate the displacement vector
        total_displacement = np.dot(di[idx] - dj[idx], latvecs)
        # Add the atomic positions
        atom_pos_i = basis[ai[idx]]
        atom_pos_j = basis[aj[idx]]
        # Calculate the relative position vector
        relative_position = atom_pos_j + total_displacement - atom_pos_i
        
        # Calculate the distance and out-of-plane distance
        atom_distance = np.linalg.norm(relative_position)
        out_of_plane_distance = np.abs(relative_position[2])
        
        # Compute the hopping using the custom function
        hoppings[idx] = compute_hopping(atom_distance, out_of_plane_distance)

    return hoppings


try:
    targets = process_hdf5_to_tuple('75.2', 1)
    target = targets[0]
    conversion = 1.0/.529177 # convert angstrom to bohr radius
    a = 2.46 # graphene lattice constant in angstrom
    latvecs = np.array([
        [a, 0.0, 0.0],
        [-1/2*a, 3**0.5/2*a, 0.0],
        [0.0, 0.0, 30]
        ]) * conversion
    basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
    ai =  np.array([1, 1, 1, 3, 3, 3, 2, 3, 3, 3, 1, 1, 1, 1])
    aj =  np.array([0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 3, 2, 2, 2])
    di = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])
    dj = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    assert np.allclose(mk(latvecs, basis, di, dj, ai, aj), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e