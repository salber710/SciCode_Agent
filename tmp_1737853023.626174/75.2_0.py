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
    if d <= 0:
        raise ValueError("Distance d must be positive and non-zero.")
    if dz > d:
        raise ValueError("Out-of-plane distance dz cannot be greater than the total distance d.")
    if dz < 0:
        raise ValueError("Out-of-plane distance dz must be non-negative.")

    # Calculate V_ppπ using the exponential decay formula
    V_pp_pi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_ppσ using the exponential decay formula
    V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the square of the ratio of the out-of-plane distance to the total distance
    dz_over_d_squared = (dz / d) ** 2
    
    # Calculate the hopping parameter -t(R_i, R_j)
    hopping = V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared
    
    return hopping



# Background: In the context of the Moon and Koshino model, the hopping parameter between atoms in a graphene system
# is influenced by the relative positions of the atoms within the lattice. The lattice vectors define the periodic
# structure of the material, and the atomic basis specifies the positions of atoms within a unit cell. The displacement
# indices (di, dj) and atomic basis indices (ai, aj) are used to determine the specific atoms involved in the hopping
# process. The task is to compute the hopping parameter for each pair of atoms specified by these indices, using the
# previously defined hopping_mk function. This involves calculating the distance and out-of-plane distance between
# the atoms, which are necessary inputs for the hopping_mk function.


def mk(latvecs, basis, di, dj, ai, aj):
    '''Evaluate the Moon and Koshino hopping parameters Phys. Rev. B 85, 195458 (2012).
    Args:
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr; natoms: number of atoms within a unit cell
        di, dj (np.array): list of displacement indices for the hopping
        ai, aj (np.array): list of atomic basis indices for the hopping
    Return
        hopping (np.array): a list with the same length as di
    '''
    def hopping_mk(d, dz, v_p0=-2.7, v_s0=0.48, b=1.17, a0=2.68, d0=6.33):
        if d <= 0:
            raise ValueError("Distance d must be positive and non-zero.")
        if dz > d:
            raise ValueError("Out-of-plane distance dz cannot be greater than the total distance d.")
        if dz < 0:
            raise ValueError("Out-of-plane distance dz must be non-negative.")

        V_pp_pi = v_p0 * np.exp(-b * (d - a0))
        V_pp_sigma = v_s0 * np.exp(-b * (d - d0))
        dz_over_d_squared = (dz / d) ** 2
        hopping = V_pp_pi * (1 - dz_over_d_squared) + V_pp_sigma * dz_over_d_squared
        return hopping

    hopping = []
    for i in range(len(di)):
        # Calculate the displacement vector between the two atoms
        displacement = np.dot(di[i] - dj[i], latvecs) + (basis[ai[i]] - basis[aj[i]])
        
        # Calculate the total distance and the out-of-plane distance
        d = np.linalg.norm(displacement)
        dz = np.abs(displacement[2])  # z-component is the out-of-plane distance
        
        # Calculate the hopping parameter using the hopping_mk function
        hop_value = hopping_mk(d, dz)
        hopping.append(hop_value)
    
    return np.array(hopping)

from scicode.parse.parse import process_hdf5_to_tuple
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
