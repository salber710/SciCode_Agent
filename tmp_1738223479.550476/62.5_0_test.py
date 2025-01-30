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


def block_enlarged(block, model_d):
    '''Enlarges the given quantum block by one unit and updates its operators.'''



    # Extract the current block's Hamiltonian and connection operators
    H_b = block.operator_dict["H"]
    Sz_b = block.operator_dict["conn_Sz"]
    Sp_b = block.operator_dict["conn_Sp"]

    # Define single-site spin operators for the new site using a diagonal approach
    Sz_new_diag = diags([0.5, -0.5])
    Sp_new_diag = diags([0, 1], offsets=-1)

    # Create identity matrices using BSR format
    identity_block_bsr = bsr_matrix((np.ones(block.basis_size), np.arange(block.basis_size)), shape=(block.basis_size, block.basis_size))
    identity_site_bsr = bsr_matrix((np.ones(model_d), np.arange(model_d)), shape=(model_d, model_d))

    # Calculate the new Hamiltonian for the enlarged system with a unique use of BSR and diagonal matrices
    H_e = kron(H_b, identity_site_bsr, format='bsr') + kron(identity_block_bsr, Sz_new_diag, format='bsr')

    # Include interaction terms using a novel method with diagonals
    interaction_sz = kron(Sz_b, Sz_new_diag, format='bsr')
    interaction_sp = kron(Sp_b, Sp_new_diag, format='bsr') + kron(Sp_b.getH(), Sp_new_diag.conj().T, format='bsr')
    H_e += interaction_sz + interaction_sp

    # Construct the new connection operators for the enlarged block using BSR
    conn_Sz_e = kron(identity_block_bsr, Sz_new_diag, format='bsr')
    conn_Sp_e = kron(identity_block_bsr, Sp_new_diag, format='bsr')

    # Update the operator dictionary for the enlarged block
    operator_dict_e = {
        "H": H_e,
        "conn_Sz": conn_Sz_e,
        "conn_Sp": conn_Sp_e
    }

    # Create an instance of the EnlargedBlock class with updated attributes
    eblock = EnlargedBlock(length=block.length + 1, basis_size=block.basis_size * model_d, operator_dict=operator_dict_e)

    return eblock



# Background: In the DMRG (Density Matrix Renormalization Group) algorithm, we iteratively construct larger 
# quantum systems by adding sites and then reducing the dimensionality of the system's description to keep 
# calculations manageable. The superblock is the combination of the system and environment blocks, each 
# enlarged by one site. The Hamiltonian of the superblock describes the full interactions within this 
# combined system. We compute the reduced density matrix of the enlarged system, whose eigenvectors are used 
# to construct a transformation matrix. This matrix reduces the basis size of the system block to a target 
# dimension while retaining the most significant features of the quantum state. The reduced basis is defined 
# by the eigenvectors corresponding to the largest eigenvalues of the reduced density matrix.

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
    np.random.seed(42)  # For reproducibility

    # Enlarge system and environment blocks
    sys_enlarged = block_enlarged(sys, model_d)
    env_enlarged = block_enlarged(env, model_d)

    # Extract operators from enlarged system and environment
    H_sys = sys_enlarged.operator_dict["H"]
    H_env = env_enlarged.operator_dict["H"]
    Sz_sys = sys_enlarged.operator_dict["conn_Sz"]
    Sp_sys = sys_enlarged.operator_dict["conn_Sp"]
    Sz_env = env_enlarged.operator_dict["conn_Sz"]
    Sp_env = env_enlarged.operator_dict["conn_Sp"]
    
    # Construct the Hamiltonian for the superblock
    H_univ = kron(H_sys, identity(env_enlarged.basis_size)) + \
             kron(identity(sys_enlarged.basis_size), H_env) + \
             kron(Sz_sys, Sz_env) + \
             0.5 * (kron(Sp_sys, Sp_env.conj().T) + kron(Sp_sys.conj().T, Sp_env))

    # Compute the ground state of the superblock Hamiltonian using eigsh
    v0 = np.random.rand(H_univ.shape[0])  # Initial vector for eigsh
    energy, psi_ground = eigsh(H_univ, k=1, which='SA', v0=v0, maxiter=10000)
    energy = energy[0]  # Ground state energy

    # Reshape ground state wavefunction into a matrix
    psi_matrix = psi_ground.reshape((sys_enlarged.basis_size, env_enlarged.basis_size))

    # Compute the reduced density matrix for the enlarged system
    rho_sys = psi_matrix @ psi_matrix.conj().T

    # Diagonalize the reduced density matrix to find eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(rho_sys)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Determine the number of states to keep (min(m, m0 * D))
    m0_prime_D = env_enlarged.basis_size
    m_tilde = min(m, m0_prime_D)

    # Construct the transformation matrix O using the top m_tilde eigenvectors
    O = eigenvectors[:, :m_tilde]

    # Transform the operators of the enlarged system block to the reduced basis
    H_new = O.conj().T @ H_sys @ O
    conn_Sz_new = O.conj().T @ Sz_sys @ O
    conn_Sp_new = O.conj().T @ Sp_sys @ O

    # Construct the new operator dictionary
    operator_dict_new = {
        "H": H_new,
        "conn_Sz": conn_Sz_new,
        "conn_Sp": conn_Sp_new
    }

    # Create the new block with reduced basis
    newblock = Block(length=sys_enlarged.length, basis_size=m_tilde, operator_dict=operator_dict_new)

    return newblock, energy


try:
    targets = process_hdf5_to_tuple('62.5', 1)
    target = targets[0]
    from scicode.compare.cmp import are_dicts_close
    model_d = 2
    block = block_initial(model_d)
    sys = block
    env = block
    m = 10
    newblock, energy = dmrg_module(sys, env, m, model_d)
    a, b, c, d = target
    assert np.allclose(newblock.length, a) and np.allclose(newblock.basis_size, b) and are_dicts_close(newblock.operator_dict, c) and np.allclose(energy, d)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e