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

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('62.2', 1)
target = targets[0]

from scicode.compare.cmp import are_dicts_close
model_d = 2
block = block_initial(model_d)
a, b, c = target
assert np.allclose(block.length, a) and np.allclose(block.basis_size, b) and are_dicts_close(block.operator_dict, c)
