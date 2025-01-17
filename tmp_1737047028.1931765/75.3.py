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


# Background: In the context of the Moon and Koshino model, the hopping parameter between atoms in a graphene lattice
# is influenced by the relative positions of the atoms within the lattice. The lattice vectors define the periodicity
# and orientation of the lattice, while the basis provides the specific atomic positions within a unit cell. The 
# displacement indices (di, dj) and atomic basis indices (ai, aj) are used to determine the specific pairs of atoms 
# for which the hopping parameter is to be calculated. The hopping parameter is then evaluated using the previously 
# defined function, which takes into account the distance and orientation of the atomic orbitals.


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
    # Initialize an array to store the hopping values
    hopping = np.zeros(len(di))
    
    # Iterate over each pair of displacement and atomic basis indices
    for idx in range(len(di)):
        # Calculate the displacement vector using the lattice vectors and displacement indices
        displacement = np.dot(di[idx] - dj[idx], latvecs)
        
        # Calculate the position of the two atoms using the basis and atomic indices
        pos_i = basis[ai[idx]] + displacement
        pos_j = basis[aj[idx]]
        
        # Calculate the distance vector between the two atoms
        d_vec = pos_i - pos_j
        
        # Calculate the total distance d and the out-of-plane distance dz
        d = np.linalg.norm(d_vec)
        dz = np.abs(d_vec[2])  # Assuming the z-component is the out-of-plane component
        
        # Use the hopping_mk function to calculate the hopping parameter
        hopping[idx] = hopping_mk(d, dz)
    
    return hopping



# Background: In solid-state physics, the Hamiltonian matrix at a given k-point in the Brillouin zone
# is used to describe the electronic structure of a crystal. The k-point, given in reduced coordinates,
# represents a point in reciprocal space. The Hamiltonian matrix is constructed using the hopping 
# parameters between atoms, which are influenced by the lattice vectors and atomic basis positions.
# The eigenvalues of this matrix correspond to the energy levels of the system at that k-point. 
# Calculating these eigenvalues and sorting them provides insight into the band structure of the material.


def ham_eig(k_input, latvecs, basis):
    '''Calculate the eigenvalues for a given k-point (k-point is in reduced coordinates)
    Args:
        k_input (np.array): (kx, ky)
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr
    Returns:
        eigval: numpy array of floats, sorted array of eigenvalues
    '''
    # Number of atoms in the basis
    natoms = basis.shape[0]
    
    # Initialize the Hamiltonian matrix
    H = np.zeros((natoms, natoms), dtype=complex)
    
    # Iterate over each pair of atoms in the basis
    for i in range(natoms):
        for j in range(natoms):
            # Calculate the phase factor for the k-point
            phase_factor = np.exp(1j * np.dot(k_input, basis[i] - basis[j]))
            
            # Calculate the hopping parameter using the mk function
            # Here we assume mk function is available and returns the hopping parameter
            # For simplicity, we assume di, dj, ai, aj are such that they correspond to i, j
            # In practice, these would be determined based on the lattice and basis
            hopping = mk(latvecs, basis, np.array([0]), np.array([0]), np.array([i]), np.array([j]))[0]
            
            # Fill the Hamiltonian matrix
            H[i, j] = hopping * phase_factor
    
    # Calculate the eigenvalues of the Hamiltonian matrix
    eigval = np.linalg.eigvalsh(H)
    
    # Sort the eigenvalues in ascending order
    eigval = np.sort(eigval)
    
    return eigval


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('75.3', 3)
target = targets[0]

k = np.array([0.5, 0.0])
# test system
conversion = 1.0/.529177 # convert angstrom to bohr radius
a = 2.46 # graphene lattice constant in angstrom
latvecs = np.array([
    [a, 0.0, 0.0],
    [-1/2*a, 3**0.5/2*a, 0.0],
    [0.0, 0.0, 30]
    ]) * conversion
basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
assert np.allclose(ham_eig(k, latvecs, basis), target)
target = targets[1]

k = np.array([0.0, 0.0])
# test system
conversion = 1.0/.529177 # convert angstrom to bohr radius
a = 2.46 # graphene lattice constant in angstrom
latvecs = np.array([
    [a, 0.0, 0.0],
    [-1/2*a, 3**0.5/2*a, 0.0],
    [0.0, 0.0, 30]
    ]) * conversion
basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
assert np.allclose(ham_eig(k, latvecs, basis), target)
target = targets[2]

k = np.array([2/3, -1/3])
# test system
conversion = 1.0/.529177 # convert angstrom to bohr radius
a = 2.46 # graphene lattice constant in angstrom
latvecs = np.array([
    [a, 0.0, 0.0],
    [-1/2*a, 3**0.5/2*a, 0.0],
    [0.0, 0.0, 30]
    ]) * conversion
basis = np.array([[0, 0, 0], [0, 1/3**0.5*a, 0], [0, 0, 3.44], [0, 1/3**0.5*a, 3.44]]) * conversion
assert np.allclose(ham_eig(k, latvecs, basis), target)
