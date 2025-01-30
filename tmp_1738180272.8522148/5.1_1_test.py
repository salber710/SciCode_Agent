from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




def lanczos(A, b, m):
    M = A.shape[0]
    Q = np.zeros((M, m+1))
    Q[:, 0] = b / np.linalg.norm(b)
    
    for i in range(1, m+1):
        v = A @ Q[:, i-1]
        for j in range(i):
            Q[:, i] += np.dot(Q[:, j], v) * Q[:, j]
        v -= Q[:, i]
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            Q[:, i] = v / v_norm
    
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