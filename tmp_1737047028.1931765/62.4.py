import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
def __init__(self, length, basis_size, operator_dict):
    self.length = length
    self.basis_size = basis_size
    self.operator_dict = operator_dict

# Background: 
# The Heisenberg XXZ model is a quantum spin model that describes interactions between spins on a lattice. 
# In the 1D spin-1/2 Heisenberg XXZ model, each site can be in one of two states: spin up (|↑⟩) or spin down (|↓⟩).
# The spin-z operator, denoted as S^z, measures the spin component along the z-axis and is represented by a diagonal matrix:
# S^z = [[1/2, 0], [0, -1/2]].
# The spin ladder operator S^+ raises the spin state from |↓⟩ to |↑⟩ and is represented by:
# S^+ = [[0, 1], [0, 0]].
# The single-site Hamiltonian for the XXZ model without an external magnetic field is given by:
# H_1 = J * (S^x ⊗ S^x + S^y ⊗ S^y + Δ * S^z ⊗ S^z), where J is the exchange interaction (set to 1 here), 
# and Δ is the anisotropy parameter (set to 1 for isotropic interactions).
# For a single site, the Hamiltonian simplifies to H_1 = 0 because there are no interactions with other sites.


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
    # Define the spin-z operator S^z
    Sz1 = np.array([[0.5, 0], [0, -0.5]])
    
    # Define the spin ladder operator S^+
    Sp1 = np.array([[0, 1], [0, 0]])
    
    # For a single site, the Hamiltonian H_1 is zero because there are no interactions
    H1 = np.zeros((model_d, model_d))
    
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
# In the Heisenberg XXZ model, we consider interactions between spins on a lattice. 
# For a two-site system, the Hamiltonian is constructed using the tensor product of operators from each site.
# The Hamiltonian for the two-site Heisenberg XXZ model is given by:
# H_2 = J * (S^x_1 ⊗ S^x_2 + S^y_1 ⊗ S^y_2 + Δ * S^z_1 ⊗ S^z_2), where J is the exchange interaction (set to 1 here),
# and Δ is the anisotropy parameter (set to 1 for isotropic interactions).
# The spin operators S^x and S^y can be expressed in terms of the ladder operators S^+ and S^-:
# S^x = (S^+ + S^-)/2 and S^y = (S^+ - S^-)/(2i).
# The spin lowering operator S^- is the Hermitian conjugate of S^+.
# For the two-site Hamiltonian, we use the Kronecker product to combine operators from each site.




def H_XXZ(Sz1, Sp1, Sz2, Sp2):
    '''Constructs the two-site Heisenberg XXZ chain Hamiltonian in matrix form.
    Input:
    Sz1,Sz2: 2d array of float, spin-z operator on site 1(or 2)
    Sp1,Sp2: 2d array of float, spin ladder operator on site 1(or 2)
    Output:
    H2_mat: sparse matrix of float, two-site Heisenberg XXZ chain Hamiltonian
    '''
    # Define the spin lowering operator S^- as the Hermitian conjugate of S^+
    Sm1 = Sp1.conj().T
    Sm2 = Sp2.conj().T

    # Define the spin-x and spin-y operators using the ladder operators
    Sx1 = 0.5 * (Sp1 + Sm1)
    Sx2 = 0.5 * (Sp2 + Sm2)
    Sy1 = -0.5j * (Sp1 - Sm1)
    Sy2 = -0.5j * (Sp2 - Sm2)

    # Construct the two-site Hamiltonian using Kronecker products
    H2_mat = (kron(Sx1, Sx2) + kron(Sy1, Sy2) + kron(Sz1, Sz2))

    return H2_mat



# Background: 
# In the Density Matrix Renormalization Group (DMRG) algorithm, we iteratively enlarge a block by adding new sites.
# When a block is enlarged, the Hamiltonian of the enlarged block is constructed by considering the interactions
# between the existing block and the new site. The connection operators for the enlarged block are defined using
# the Kronecker product of the identity operator for the block and the spin operators for the new site.
# The Hamiltonian for the enlarged block is constructed using the Kronecker product of the block's Hamiltonian
# and the identity operator for the new site, plus the interaction Hamiltonian between the block and the new site.
# The connection operators for the enlarged block are similarly constructed using the Kronecker product.

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
    # Extract the current block's attributes
    l = block.length
    m = block.basis_size
    H_b = block.operator_dict["H"]
    Sz_b = block.operator_dict["conn_Sz"]
    Sp_b = block.operator_dict["conn_Sp"]

    # Define the spin-z and spin ladder operators for the new site
    Sz_new = np.array([[0.5, 0], [0, -0.5]])
    Sp_new = np.array([[0, 1], [0, 0]])

    # Construct the identity operator for the block
    I_b = identity(m)

    # Construct the new Hamiltonian for the enlarged block
    H_e = kron(H_b, identity(model_d)) + H_XXZ(Sz_b, Sp_b, Sz_new, Sp_new)

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
    eblock = EnlargedBlock(length=l+1, basis_size=m*model_d, operator_dict=operator_dict)

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
