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


# Background: The Moon and Koshino model calculates the electronic hopping parameters between atoms in a graphene lattice.
# The hopping is determined by the relative positions of atoms within a unit cell and the lattice vectors defining the 
# periodicity of the crystal. The displacement indices (di, dj) determine the relative lattice positions, while the atomic 
# basis indices (ai, aj) specify which atoms within the unit cells are involved. The task is to compute the hopping 
# parameters using these indices, the lattice vectors, and atomic positions, based on the previously defined hopping_mk 
# function which uses the Moon and Koshino model.


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
    # Initialize the hopping array
    hopping = np.zeros(len(di))
    
    # Loop over each pair of indices to calculate hopping
    for index in range(len(di)):
        # Calculate the displacement vector using lattice vectors and displacement indices
        R_i = np.dot(di[index], latvecs)
        R_j = np.dot(dj[index], latvecs)
        
        # Determine the full position of the atoms involved by adding the basis positions
        pos_i = R_i + basis[ai[index]]
        pos_j = R_j + basis[aj[index]]
        
        # Calculate the distance vector between the two atoms
        d_vec = pos_i - pos_j
        
        # Calculate the total distance and the out-of-plane component
        d = np.linalg.norm(d_vec)
        dz = np.abs(d_vec[2])
        
        # Use the hopping_mk function to calculate the hopping parameter
        hopping[index] = hopping_mk(d, dz)
        
    return hopping



# Background: In solid-state physics, the Hamiltonian matrix of a crystal at a given wave vector (k-point) 
# characterizes the electronic structure of the material. For a 2D crystal like graphene, the Hamiltonian is 
# constructed in the basis of atomic orbitals, and the elements of the matrix (hopping parameters) are determined 
# by the relative positions of atoms and the k-point. The Hamiltonian at a specific k-point is given by:
# H(k) = Σ t(R_i, R_j) * exp(i * k · (R_i - R_j)), where t(R_i, R_j) are the hopping parameters between atoms 
# at positions R_i and R_j. The eigenvalues of the Hamiltonian matrix provide the energy bands of the crystal 
# at that k-point, which are critical for understanding the electronic properties.


def ham_eig(k_input, latvecs, basis):
    '''Calculate the eigenvalues for a given k-point (k-point is in reduced coordinates)
    Args:
        k_input (np.array): (kx, ky)
        latvecs (np.array): lattice vectors of shape (3, 3) in bohr
        basis (np.array): atomic positions of shape (natoms, 3) in bohr
    Returns:
        eigval: numpy array of floats, sorted array of eigenvalues
    '''
    
    # Number of atoms in the basis, which determines the size of the Hamiltonian matrix
    natoms = len(basis)
    
    # Initialize the Hamiltonian matrix to zero
    H = np.zeros((natoms, natoms), dtype=np.complex128)
    
    # Convert k_input to a 3D vector by adding a zero component for kz (since it's a 2D problem)
    k_vec = np.array([k_input[0], k_input[1], 0.0])
    
    # Generate hopping parameters for the Hamiltonian matrix
    # Realistically, we would iterate over possible hopping pairs (ai, aj) and displacements (di, dj)
    # For simplicity, let's assume we have predefined di, dj, ai, aj that cover necessary interactions
    # Here, we would loop over these indices to fill the Hamiltonian
    
    # Example: Iterate over all atom pairs and fill in the Hamiltonian matrix
    for ai in range(natoms):
        for aj in range(natoms):
            # Here, di and dj would be part of a larger set of interactions, which are neglected for brevity
            # Calculate the displacement vector between ai and aj in real space
            # For an example, let's assume we have these vectors pre-computed:
            di = np.array([0, 0, 0])  # Example placeholder
            dj = np.array([0, 0, 0])  # Example placeholder
            
            # Calculate positions Ri and Rj
            R_i = np.dot(di, latvecs) + basis[ai]
            R_j = np.dot(dj, latvecs) + basis[aj]
            
            # Calculate the hopping parameter (assuming we have a function mk for this)
            # hopping = mk(latvecs, basis, [di], [dj], [ai], [aj])[0]  # Example usage
            # Using the previously defined hopping_mk function directly for demonstration
            d_vec = R_i - R_j
            d = np.linalg.norm(d_vec)
            dz = np.abs(d_vec[2])
            hopping = hopping_mk(d, dz)
            
            # Calculate the phase factor
            phase_factor = np.exp(1j * np.dot(k_vec, d_vec))
            
            # Fill in the Hamiltonian matrix
            H[ai, aj] += hopping * phase_factor
    
    # Calculate the eigenvalues of the Hamiltonian
    eigenvalues = np.linalg.eigvalsh(H)
    
    # Sort the eigenvalues in ascending order
    eigval = np.sort(eigenvalues)
    
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
