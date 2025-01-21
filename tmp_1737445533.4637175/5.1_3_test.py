try:
    import numpy as np
    
    
    
    # Background: 
    # The Lanczos Iteration is an algorithm used to reduce a large symmetric matrix to a tridiagonal form, which is useful in numerical linear algebra for efficiently computing eigenvalues and eigenvectors. 
    # The process involves constructing a sequence of orthonormal vectors that form the columns of a matrix Q, and a tridiagonal matrix T, such that A is approximately equal to QTQ^T.
    # The algorithm begins with an initial vector 'b', which is typically a random vector, and iterates to produce an orthonormal basis for the Krylov subspace spanned by {b, Ab, A^2b, ..., A^(m-1)b}.
    # Orthogonalization is performed using the Gram-Schmidt process to ensure that the vectors remain orthogonal.
    # The output Q has orthonormal columns, and the first m columns of Q represent the desired orthonormal basis.
    
    
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
        beta = np.zeros(m+1)
    
        # Normalize the initial vector b
        Q[:, 0] = b / np.linalg.norm(b)
    
        for j in range(m):
            if j == 0:
                v = A @ Q[:, j]
            else:
                v = A @ Q[:, j] - beta[j] * Q[:, j-1]
            
            alpha[j] = np.dot(Q[:, j], v)
            v = v - alpha[j] * Q[:, j]
            
            # Re-orthogonalize (if necessary, to ensure numerical stability)
            for i in range(j+1):
                v = v - np.dot(Q[:, i], v) * Q[:, i]
            
            beta[j+1] = np.linalg.norm(v)
            
            if beta[j+1] != 0:
                Q[:, j+1] = v / beta[j+1]
            else:
                # If the norm is zero, stop the iteration as it indicates convergence
                break
    
        return Q
    

    from scicode.parse.parse import process_hdf5_to_tuple
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
