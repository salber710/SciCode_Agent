import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
def __init__(self, length, basis_size, operator_dict):
    self.length = length
    self.basis_size = basis_size
    self.operator_dict = operator_dict

# Background: 
# The Heisenberg XXZ model is a quantum spin model used to describe interactions between spins on a lattice. 
# In the 1D spin-1/2 Heisenberg XXZ model, each site can be in one of two states: spin up (|↑⟩) or spin down (|↓⟩).
# The spin-z operator, denoted as Sz, measures the z-component of the spin. For a single spin-1/2 particle, 
# Sz is represented by the matrix: Sz = 0.5 * [[1, 0], [0, -1]].
# The spin ladder operators, S+ and S-, are used to raise or lower the spin state. The raising operator S+ is 
# represented by the matrix: S+ = [[0, 1], [0, 0]].
# The Hamiltonian for a single site in the XXZ model without an external magnetic field is given by the matrix:
# H1 = 0.5 * [[0, 0], [0, 0]], which is essentially zero for a single site as there are no interactions.
# In this step, we construct the initial block for the DMRG algorithm using these operators.


class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

def block_initial(model_d):
    '''Construct the initial block for the DMRG algo. H1, Sz1 and Sp1 is single-site Hamiltonian, spin-z operator
    and spin ladder operator in the form of 2x2 matrix, respectively.
    Input:
    model_d: int, single-site basis size
    Output:
    initial_block: instance of the "Block" class, with attributes "length", "basis_size", "operator_dict"
                  - length: An integer representing the block's current length.
                  - basis_size: An integer indicating the size of the basis.
                  - operator_dict: A dictionary containing operators: Hamiltonian ("H":H1), 
                                   Connection operator ("conn_Sz":Sz1), and Connection operator("conn_Sp":Sp1).
                                   H1, Sz1 and Sp1: 2d array of float
    '''
    if not isinstance(model_d, int):
        raise TypeError("model_d must be an integer")
    if model_d <= 0:
        raise ValueError("model_d must be a positive integer")
    
    # Define the spin-z operator Sz for a single spin-1/2 particle
    Sz1 = 0.5 * np.array([[1, 0], [0, -1]])
    
    # Define the spin ladder operator S+ for a single spin-1/2 particle
    Sp1 = np.array([[0, 1], [0, 0]])
    
    # Define the Hamiltonian H1 for a single site in the XXZ model
    H1 = 0.5 * np.array([[0, 0], [0, 0]])
    
    # Create the operator dictionary
    operator_dict = {
        "H": H1,
        "conn_Sz": Sz1,
        "conn_Sp": Sp1
    }
    
    # Create an instance of the Block class
    initial_block = Block(length=1, basis_size=model_d, operator_dict=operator_dict)
    
    return initial_block


# Background: 
# In the Heisenberg XXZ model, we are interested in interactions between spins on a lattice. 
# For a two-site system, the Hamiltonian can be constructed using the spin operators for each site.
# The Hamiltonian for the two-site Heisenberg XXZ model is given by:
# H = J * (S1^x S2^x + S1^y S2^y + Δ S1^z S2^z)
# where S1 and S2 are the spin operators for site 1 and site 2, respectively, and Δ is the anisotropy parameter.
# For the isotropic case (Δ = 1), the Hamiltonian simplifies to:
# H = S1^x S2^x + S1^y S2^y + S1^z S2^z
# The spin-x and spin-y operators can be expressed in terms of the ladder operators:
# S^x = 0.5 * (S^+ + S^-)
# S^y = 0.5 * (S^+ - S^-)
# The Kronecker product is used to construct the Hamiltonian for the two-site system from the single-site operators.



def H_XXZ(Sz1, Sp1, Sz2, Sp2, J=1.0, Delta=1.0):
    '''Constructs the two-site Heisenberg XXZ chain Hamiltonian in matrix form.
    Input:
    Sz1,Sz2: 2d array of float, spin-z operator on site 1(or 2)
    Sp1,Sp2: 2d array of float, spin ladder operator on site 1(or 2)
    J: float, coupling constant
    Delta: float, anisotropy parameter
    Output:
    H2_mat: sparse matrix of float, two-site Heisenberg XXZ chain Hamiltonian
    '''
    # Define the spin-x and spin-y operators using the ladder operators
    Sx1 = 0.5 * (Sp1 + Sp1.T.conj())
    Sy1 = 0.5j * (Sp1 - Sp1.T.conj())
    Sx2 = 0.5 * (Sp2 + Sp2.T.conj())
    Sy2 = 0.5j * (Sp2 - Sp2.T.conj())
    
    # Construct the two-site Hamiltonian using the Kronecker product
    H2_mat = J * (kron(Sx1, Sx2) + kron(Sy1, Sy2) + Delta * kron(Sz1, Sz2))
    
    return csr_matrix(H2_mat)


# Background: 
# In the Density Matrix Renormalization Group (DMRG) algorithm, we iteratively enlarge a block by adding new sites.
# When a block is enlarged, the Hamiltonian of the system must be updated to include interactions between the block
# and the new site. The connection operators for the enlarged block are defined using the Kronecker product, which
# allows us to express the operators in the combined basis of the block and the new site.
# The Hamiltonian for the enlarged block is constructed using the Hamiltonian of the original block and the new site,
# as well as the interaction terms between them. The connection operators for the enlarged block are similarly updated
# to reflect the new basis.



class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

def block_enlarged(block, model_d):
    '''Enlarges the given quantum block by one unit and updates its operators.
    Input:
    - block: instance of the "Block" class with the following attributes:
      - length: An integer representing the block's current length.
      - basis_size: An integer representing the size of the basis associated with the block.
      - operator_dict: A dictionary of quantum operators for the block:
          - "H": The Hamiltonian of the block.
          - "conn_Sz": A connection matrix, if length is 1, it corresponds to the spin-z operator.
          - "conn_Sp": A connection matrix, if length is 1, it corresponds to the spin ladder operator.
    - model_d: int, single-site basis size
    Output:
    - eblock: instance of the "Block" class with the following attributes:
      - length: An integer representing the new length.
      - basis_size: An integer representing the new size of the basis.
      - operator_dict: A dictionary of updated quantum operators:
          - "H": An updated Hamiltonian matrix of the enlarged system.
          - "conn_Sz": A new connection matrix.
          - "conn_Sp": Another new connection matrix.
    '''
    if model_d <= 0:
        raise ValueError("Model dimension must be greater than zero.")

    # Extract the current block's Hamiltonian and connection operators
    H_b = block.operator_dict["H"]
    Sz_b = block.operator_dict["conn_Sz"]
    Sp_b = block.operator_dict["conn_Sp"]
    
    # Define the single-site operators for the new site
    Sz_new = 0.5 * np.array([[1, 0], [0, -1]])
    Sp_new = np.array([[0, 1], [0, 0]])
    
    # Create identity matrices for the Kronecker product
    I_b = identity(block.basis_size, format='csr')
    I_new = identity(model_d, format='csr')
    
    # Construct the enlarged Hamiltonian using the Kronecker product
    H_e = kron(H_b, I_new, format='csr') + kron(I_b, 0.5 * np.array([[0, 0], [0, 0]]), format='csr') + \
          kron(Sz_b, Sz_new, format='csr') + 0.5 * (kron(Sp_b, Sp_new, format='csr') + kron(Sp_b.T.conj(), Sp_new.T.conj(), format='csr'))
    
    # Construct the new connection operators for the enlarged block
    Sz_e = kron(I_b, Sz_new, format='csr')
    Sp_e = kron(I_b, Sp_new, format='csr')
    
    # Create the operator dictionary for the enlarged block
    operator_dict = {
        "H": H_e,
        "conn_Sz": Sz_e,
        "conn_Sp": Sp_e
    }
    
    # Create an instance of the Block class for the enlarged block
    eblock = Block(length=block.length + 1, basis_size=block.basis_size * model_d, operator_dict=operator_dict)
    
    return eblock


# Background: 
# In the Density Matrix Renormalization Group (DMRG) algorithm, we iteratively optimize the representation of a quantum system by enlarging blocks and constructing a superblock. 
# The superblock Hamiltonian, H_univ, is constructed by combining the Hamiltonians of the enlarged system and environment blocks, including interactions between them.
# The reduced density matrix, rho_sys, is computed for the enlarged system to capture the most significant states.
# The transformation matrix, O, is derived from the eigenvectors of rho_sys corresponding to the largest eigenvalues, allowing us to truncate the basis to a target dimension, m.
# This process helps in efficiently representing the quantum state by focusing on the most relevant degrees of freedom.




class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

def block_enlarged(block, model_d):
    # Enlarge the block by creating new operators with increased basis size
    new_basis_size = block.basis_size * model_d
    new_operators = {}
    for key, op in block.operator_dict.items():
        new_operators[key] = kron(op, identity(model_d, format='csr'), format='csr')
    return Block(length=block.length + 1, basis_size=new_basis_size, operator_dict=new_operators)

def dmrg_module(sys, env, m, model_d):
    '''Input:
    sys: instance of the "Block" class
    env: instance of the "Block" class
    m: int, number of states in the new basis, i.e. the dimension of the new basis
    model_d: int, single-site basis size
    Output:
    newblock: instance of the "Block" class
    energy: superblock ground state energy, float
    '''
    if m <= 0:
        raise ValueError("The number of states 'm' must be positive.")
    if model_d <= 0:
        raise ValueError("The model dimension 'model_d' must be positive.")
    if sys.basis_size <= 0 or env.basis_size <= 0:
        raise ValueError("Basis size must be positive.")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Enlarge the system and environment blocks
    sys_enlarged = block_enlarged(sys, model_d)
    env_enlarged = block_enlarged(env, model_d)

    # Extract operators from the enlarged blocks
    H_sys = sys_enlarged.operator_dict["H"]
    Sz_sys = sys_enlarged.operator_dict["conn_Sz"]
    Sp_sys = sys_enlarged.operator_dict["conn_Sp"]

    H_env = env_enlarged.operator_dict["H"]
    Sz_env = env_enlarged.operator_dict["conn_Sz"]
    Sp_env = env_enlarged.operator_dict["conn_Sp"]

    # Create identity matrices for Kronecker products
    I_sys = identity(sys_enlarged.basis_size, format='csr')
    I_env = identity(env_enlarged.basis_size, format='csr')

    # Construct the superblock Hamiltonian
    H_univ = kron(H_sys, I_env, format='csr') + kron(I_sys, H_env, format='csr') + \
             kron(Sz_sys, Sz_env, format='csr') + 0.5 * (kron(Sp_sys, Sp_env, format='csr') + kron(Sp_sys.T.conj(), Sp_env.T.conj(), format='csr'))

    # Compute the ground state of the superblock using eigsh
    v0 = np.random.rand(H_univ.shape[0])
    energy, psi = eigsh(H_univ, k=1, which='SA', v0=v0, return_eigenvectors=True)

    # Reshape the ground state wavefunction to a matrix
    psi_matrix = psi.reshape((sys_enlarged.basis_size, env_enlarged.basis_size))

    # Compute the reduced density matrix for the system
    rho_sys = psi_matrix @ psi_matrix.T.conj()

    # Diagonalize the reduced density matrix
    eigenvalues, eigenvectors = np.linalg.eigh(rho_sys.toarray())

    # Sort eigenvalues and select the largest ones
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    O = eigenvectors[:, :min(m, env_enlarged.basis_size)]

    # Update the system operators using the transformation matrix O
    H_new = O.T.conj() @ H_sys.toarray() @ O
    Sz_new = O.T.conj() @ Sz_sys.toarray() @ O
    Sp_new = O.T.conj() @ Sp_sys.toarray() @ O

    # Create the operator dictionary for the new block
    operator_dict = {
        "H": csr_matrix(H_new),
        "conn_Sz": csr_matrix(Sz_new),
        "conn_Sp": csr_matrix(Sp_new)
    }

    # Create the new block with updated operators
    newblock = Block(length=sys_enlarged.length, basis_size=O.shape[1], operator_dict=operator_dict)

    return newblock, energy[0]



# Background: 
# The Density Matrix Renormalization Group (DMRG) algorithm is a powerful method for finding the ground state of quantum many-body systems. 
# In this step, we aim to iteratively grow the system by enlarging both the system and environment blocks until the desired system size is reached.
# The process involves enlarging the blocks, constructing a superblock, and then using the reduced density matrix to truncate the basis.
# The goal is to efficiently represent the quantum state by focusing on the most relevant degrees of freedom, which are determined by the largest eigenvalues of the reduced density matrix.
# The algorithm iterates until the system size reaches or exceeds the target size, L, and returns the ground state energy of the infinite system.

def run_dmrg(initial_block, L, m, model_d):
    '''Performs the Density Matrix Renormalization Group (DMRG) algorithm to find the ground state energy of a system.
    Input:
    - initial_block: an instance of the "Block" class with the following attributes:
        - length: An integer representing the current length of the block.
        - basis_size: An integer indicating the size of the basis.
        - operator_dict: A dictionary containing operators:
          Hamiltonian ("H"), Connection operator ("conn_Sz"), Connection operator("conn_Sp")
    - L (int): The desired system size (total length including the system and the environment).
    - m (int): The truncated dimension of the Hilbert space for eigenstate reduction.
    - model_d (int): Single-site basis size
    Output:
    - energy (float): The ground state energy of the infinite system after the DMRG steps.
    '''
    # Initialize the system and environment blocks as identical
    sys_block = initial_block
    env_block = initial_block

    # Iterate until the system size reaches or exceeds the target size L
    while sys_block.length + env_block.length < L:
        # Perform a single DMRG step
        sys_block, energy = dmrg_module(sys_block, env_block, m, model_d)
        
        # Set the environment block to be identical to the system block
        env_block = sys_block

    return energy

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('62.6', 6)
target = targets[0]

np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)
model_d = 2
block = block_initial(model_d)
assert np.allclose(run_dmrg(block, 100,10, model_d), target)
target = targets[1]

model_d = 2
block = block_initial(model_d)
assert np.allclose(run_dmrg(block, 100,20, model_d), target)
target = targets[2]

model_d = 2
block = block_initial(model_d)
assert np.allclose(run_dmrg(block, 100,100, model_d), target)
target = targets[3]

model_d = 2
block = block_initial(model_d)
assert np.allclose(run_dmrg(block, 10,100, model_d), target)
target = targets[4]

model_d = 2
block = block_initial(model_d)
assert np.allclose(run_dmrg(block, 20,100, model_d), target)
target = targets[5]

model_d = 2
block = block_initial(model_d)
assert np.allclose(run_dmrg(block, 100,100, model_d), target)
