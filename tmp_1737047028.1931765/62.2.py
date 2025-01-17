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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('62.2', 1)
target = targets[0]

from scicode.compare.cmp import are_dicts_close
model_d = 2
block = block_initial(model_d)
a, b, c = target
assert np.allclose(block.length, a) and np.allclose(block.basis_size, b) and are_dicts_close(block.operator_dict, c)
