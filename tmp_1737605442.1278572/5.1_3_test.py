from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




def lanczos(A, b, m):
    '''Inputs:
    A : Matrix, 2d array of arbitrary size M * M
    b : Vector, 1d array of arbitrary size M * 1
    m : integer, m < M
    Outputs:
    Q : Matrix, 2d array of size M*(m+1)
    '''
    # Normalize the initial vector b to get q_1
    Q = np.zeros((A.shape[0], m + 1))
    q = b / np.linalg.norm(b)
    Q[:, 0] = q
    
    # Initialize the first beta value
    beta = 0
    v = np.zeros_like(b)
    
    for j in range(1, m+1):
        # Compute v_j = A * q_j - beta_j_1 * q_j_1
        v = A @ q - beta * Q[:, j-1]
        
        # Compute alpha_j = q_j^T * v_j
        alpha = np.dot(q, v)
        
        # Re-orthogonalize v_j with respect to q_j
        v = v - alpha * q
        
        # Compute beta_j = ||v_j||
        beta = np.linalg.norm(v)
        
        if beta != 0:
            # Normalize v_j to get the new q_{j+1}
            q = v / beta
            Q[:, j] = q
        else:
            # If beta is zero, the process terminates early
            break
    
    return Q


try:
    targets = process_hdf5_to_tuple('5.1', 3)
    target = targets[0]
    n = 7
    h = 1.0/n
    diagonal = [2/h for i in range(n)]
    diagonal_up = [-1/h for i in range(n-1)]
    diagonal_down = [-1/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    b = np.array([0.1,0.1,0.0,0.1,0.0,0.1,0.1])
    m = 5
    assert np.allclose(lanczos(A,b,m), target)

    target = targets[1]
    n = 7
    h = 1.0/n
    diagonal = [1/h for i in range(n)]
    diagonal_up = [-0.9/h for i in range(n-1)]
    diagonal_down = [-0.9/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    b = np.array([0.1,10.1,0.0,0.5,0.2,0.3,0.5])
    m = 5
    assert np.allclose(lanczos(A,b,m), target)

    target = targets[2]
    n = 7
    h = 1.0/n
    diagonal = [1/h for i in range(n)]
    diagonal_up = [-9/h for i in range(n-1)]
    diagonal_down = [-9/h for i in range(n-1)]
    A = np.diag(diagonal) + np.diag(diagonal_up, 1) + np.diag(diagonal_down, -1)
    A[:, 0] = 0
    A[0, :] = 0
    A[0, 0] = 1/h
    b = np.array([0.1,0.1,0.0,10,0.0,0.1,0.1])
    m = 4
    assert np.allclose(lanczos(A,b,m), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e