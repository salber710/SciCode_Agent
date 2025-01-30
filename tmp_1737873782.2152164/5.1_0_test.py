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
    # Ensure A is a numpy array
    A = np.array(A)
    b = np.array(b)
    
    # Get the size of the matrix A
    M = A.shape[0]
    
    # Initialize the Q matrix
    Q = np.zeros((M, m + 1))
    
    # Normalize the initial vector b to form q1
    q = b / np.linalg.norm(b)
    Q[:, 0] = q
    
    # Initialize beta as zero for the first iteration
    beta = 0
    
    # Initialize the previous vector as zero vector
    q_prev = np.zeros_like(b)
    
    for j in range(m):
        # Perform the matrix-vector multiplication
        z = A @ q
        
        # Orthogonalize against the previous vector
        z = z - beta * q_prev
        
        # Calculate alpha as the dot product of the current vector with z
        alpha = np.dot(q, z)
        
        # Orthogonalize z against q
        z = z - alpha * q
        
        # Calculate beta as the norm of z
        beta = np.linalg.norm(z)
        
        # Update the previous q
        q_prev = q
        
        # Normalize z to form the next q
        if beta != 0:
            q = z / beta
            Q[:, j+1] = q
        else:
            # If beta is zero, break the loop
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