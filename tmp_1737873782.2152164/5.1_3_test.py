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
    M = A.shape[0]
    Q = np.zeros((M, m+1))
    alpha = np.zeros(m)
    beta = np.zeros(m)
    
    # Normalize the input vector b
    q0 = b / np.linalg.norm(b)
    Q[:, 0] = q0
    
    for j in range(m):
        # Compute the matrix-vector product
        z = A @ Q[:, j]
        
        # Compute alpha_j
        alpha[j] = np.dot(Q[:, j], z)
        
        # Orthogonalize against the previous vector
        if j > 0:
            z = z - beta[j-1] * Q[:, j-1]
        
        # Orthogonalize against the current vector
        z = z - alpha[j] * Q[:, j]
        
        # Compute beta_j
        beta[j] = np.linalg.norm(z)
        
        # Normalize to get the next Lanczos vector
        if j < m - 1:
            Q[:, j+1] = z / beta[j]
    
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