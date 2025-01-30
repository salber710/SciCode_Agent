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
    - eblock: instance of the "EnlargedBlock" class with the following attributes:
      - length: An integer representing the new length.
      - basis_size: An integer representing the new size of the basis.
      - operator_dict: A dictionary of updated quantum operators:
          - "H": An updated Hamiltonian matrix of the enlarged system.
          - "conn_Sz": A new connection matrix.
          - "conn_Sp": Another new connection matrix.
          They are all sparse matrix
    '''
    # Extract the current block's Hamiltonian and connection operators
    H_b = block.operator_dict["H"]
    Sz_b = block.operator_dict["conn_Sz"]
    Sp_b = block.operator_dict["conn_Sp"]
    
    # Define the single-site operators for the new site
    Sz_new = 0.5 * np.array([[1, 0], [0, -1]])
    Sp_new = np.array([[0, 1], [0, 0]])
    
    # Create identity matrices for the Kronecker product
    I_b = identity(block.basis_size)
    I_new = identity(model_d)
    
    # Construct the enlarged Hamiltonian using the Kronecker product
    H_e = kron(H_b, I_new) + kron(I_b, 0.5 * np.array([[0, 0], [0, 0]])) + \
          kron(Sz_b, Sz_new) + 0.5 * (kron(Sp_b, Sp_new) + kron(Sp_b.T.conj(), Sp_new.T.conj()))
    
    # Construct the new connection operators for the enlarged block
    Sz_e = kron(I_b, Sz_new)
    Sp_e = kron(I_b, Sp_new)
    
    # Create the operator dictionary for the enlarged block
    operator_dict = {
        "H": H_e,
        "conn_Sz": Sz_e,
        "conn_Sp": Sp_e
    }
    
    # Create an instance of the EnlargedBlock class
    eblock = Block(length=block.length + 1, basis_size=block.basis_size * model_d, operator_dict=operator_dict)
    
    return eblock

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('62.4', 1)
target = targets[0]

from scicode.compare.cmp import are_dicts_close
model_d = 2
block = block_initial(model_d)
eblock = block_enlarged(block,model_d)
a, b, c = target
assert np.allclose(eblock.length, a) and np.allclose(eblock.basis_size, b) and are_dicts_close(eblock.operator_dict, c)
