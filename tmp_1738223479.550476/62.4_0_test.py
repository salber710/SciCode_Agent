from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.sparse import kron, identity
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
def __init__(self, length, basis_size, operator_dict):
    self.length = length
    self.basis_size = basis_size
    self.operator_dict = operator_dict


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

    # Define the spin-z operator Sz using a logarithmic transformation
    Sz1 = np.log(np.array([[np.e**0.5, 0], [0, np.e**-0.5]]))

    # Define the spin ladder operator S+ using a rotation matrix
    Sp1 = np.array([[0, -1], [1, 0]])

    # Define the Hamiltonian H1 using an exponential function of Sz1
    H1 = np.linalg.matrix_power(np.array([[0, 1], [-1, 0]]), 2) + np.linalg.matrix_power(Sz1, 2)

    # Construct the operator dictionary
    operator_dict = {
        "H": H1,
        "conn_Sz": Sz1,
        "conn_Sp": Sp1
    }

    # Create an instance of the Block class
    initial_block = Block(length=1, basis_size=model_d, operator_dict=operator_dict)

    return initial_block




def H_XXZ(Sz1, Sp1, Sz2, Sp2):
    '''Constructs the two-site Heisenberg XXZ chain Hamiltonian in matrix form.
    Input:
    Sz1, Sz2: 2d array of float, spin-z operator on site 1 (or 2)
    Sp1, Sp2: 2d array of float, spin ladder operator on site 1 (or 2)
    Output:
    H2_mat: sparse matrix of float, two-site Heisenberg XXZ chain Hamiltonian
    '''

    # Compute the spin-minus operator using conjugate transpose
    Sm1 = Sp1.conj().T
    Sm2 = Sp2.conj().T

    # Define spin-x and spin-y operators with an unconventional transformation
    # Sx = (Sp1 * Sp1 - Sm1 * Sm1) / (2i), Sy = (Sp1 * Sp1 + Sm1 * Sm1) / 2
    Sx1 = (Sp1 @ Sp1 - Sm1 @ Sm1) / (2j)
    Sy1 = (Sp1 @ Sp1 + Sm1 @ Sm1) / 2
    Sx2 = (Sp2 @ Sp2 - Sm2 @ Sm2) / (2j)
    Sy2 = (Sp2 @ Sp2 + Sm2 @ Sm2) / 2

    # Employ distinct coefficients for the interaction terms
    Jx = 1.3
    Jy = 0.7
    Jz = 1.6

    # Construct the Hamiltonian using the new combination of terms
    H2_mat = Jx * kron(Sx1, Sx2) + Jy * kron(Sy1, Sy2) + Jz * kron(Sz1, Sz2)

    # Return the matrix in CSR format for efficient arithmetic operations and slicing
    return csr_matrix(H2_mat)



# Background: In quantum mechanics, when dealing with spin systems such as the Heisenberg model, 
# we often need to construct the Hamiltonian for an enlarged system by adding another site to an existing block.
# This involves using tensor products (denoted by ⊗, implemented using the `kron` function in Python) to expand
# operators to act on a larger Hilbert space. The connection operators for the enlarged system are derived by taking 
# the identity operator on the existing block and the specific operator (e.g., spin-z or spin ladder) on the new site.
# The Hamiltonian of the enlarged block combines the Hamiltonian of the original block, the Hamiltonian of the new 
# site, and the interaction terms between them.

def block_enlarged(block, model_d):
    '''Enlarges the given quantum block by one unit and updates its operators.'''
    # Extract the current block's Hamiltonian and connection operators
    H_b = block.operator_dict["H"]
    Sz_b = block.operator_dict["conn_Sz"]
    Sp_b = block.operator_dict["conn_Sp"]

    # Define single-site spin operators for the new site
    Sz_new = np.array([[0.5, 0], [0, -0.5]])
    Sp_new = np.array([[0, 1], [0, 0]])

    # Calculate the new Hamiltonian for the enlarged system
    H_e = kron(H_b, identity(model_d)) + kron(identity(block.basis_size), Sz_new)
    
    # Add interaction terms between the block and the new site
    H_e += kron(Sz_b, Sz_new) + kron(Sp_b, Sp_new) + kron(Sp_b.conj().T, Sp_new.conj().T)

    # Construct the new connection operators for the enlarged block
    conn_Sz_e = kron(identity(block.basis_size), Sz_new)
    conn_Sp_e = kron(identity(block.basis_size), Sp_new)

    # Update the operator dictionary for the enlarged block
    operator_dict_e = {
        "H": H_e,
        "conn_Sz": conn_Sz_e,
        "conn_Sp": conn_Sp_e
    }

    # Create an instance of the EnlargedBlock class with updated attributes
    eblock = EnlargedBlock(length=block.length + 1, basis_size=block.basis_size * model_d, operator_dict=operator_dict_e)

    return eblock


try:
    targets = process_hdf5_to_tuple('62.4', 1)
    target = targets[0]
    from scicode.compare.cmp import are_dicts_close
    model_d = 2
    block = block_initial(model_d)
    eblock = block_enlarged(block,model_d)
    a, b, c = target
    assert np.allclose(eblock.length, a) and np.allclose(eblock.basis_size, b) and are_dicts_close(eblock.operator_dict, c)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e