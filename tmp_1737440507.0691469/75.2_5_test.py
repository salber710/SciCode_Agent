import numpy as np

# Background: The Moon and Koshino model describes the electronic hopping between atoms in graphene. 
# The hopping parameter, -t(R_i, R_j), is derived from the interaction between atomic orbitals. 
# It depends on the π and σ bonds, characterized by V_ppπ and V_ppσ, respectively. The expressions for these 
# interactions involve exponential decay factors based on the distances between atoms. The π bond interaction 
# (V_ppπ) decreases with distance from a reference nearest-neighbor distance a0, while the σ bond interaction 
# (V_ppσ) decreases with distance from a reference interlayer distance d0. The terms dz/d and 1 - (dz/d)^2 
# account for the directionality of the bonds, where dz is the out-of-plane component of the distance vector d.

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


    # Calculate V_ppπ based on the distance d and reference distance a0
    V_pppi = v_p0 * np.exp(-b * (d - a0))
    
    # Calculate V_ppσ based on the distance d and reference distance d0
    V_ppsigma = v_s0 * np.exp(-b * (d - d0))
    
    # Calculate the ratio of the out-of-plane component to the total distance
    dz_d_ratio = dz / d
    
    # Compute the hopping parameter -t using the Moon and Koshino model
    hopping = V_pppi * (1 - dz_d_ratio**2) + V_ppsigma * dz_d_ratio**2
    
    # Return the negative of the hopping parameter as per the model
    return -hopping



# Background: In the Moon and Koshino model, the electronic hopping between atoms in graphene is determined
# by the relative positions of the atoms within the lattice. Each atom in the lattice can be indexed by
# its position in the lattice and its atomic basis index. The displacement indices (di, dj) specify the
# lattice vectors that define the relative position between two atoms, while the atomic basis indices
# (ai, aj) determine the specific atoms within the unit cell. This function computes the hopping 
# parameter for each specified pair using the lattice vectors and basis information.


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

    # Initialize an array to store the computed hopping values
    hopping = np.zeros(len(di))

    # Iterate over each pair of displacement and atomic basis indices
    for index in range(len(di)):
        # Get the displacement vectors for the given indices
        displacement_vector = latvecs[di[index]] - latvecs[dj[index]]
        
        # Get the positions of the atoms in the basis
        position_i = basis[ai[index]]
        position_j = basis[aj[index]]
        
        # Calculate the vector connecting the two atoms
        d_vector = displacement_vector + position_i - position_j
        
        # Calculate the total distance and the out-of-plane component
        d = np.linalg.norm(d_vector)
        dz = d_vector[2]  # Assuming the third component is the out-of-plane direction
        
        # Use the hopping_mk function to calculate the hopping parameter
        hopping[index] = hopping_mk(d, dz)
        
    return hopping

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
