from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of element
    p: array of arbitrary size 1,2, or 3, the coordinates
    M: int, the total number of the nodal dofs
    h: int, the element size
    etype: int, basis function type; When type equals to 1, 
    it returns \omega^1(x), when the type equals to 2, it returns the value of function \omega^2(x).
    Outputs
    v: array of size 1,2, or 3, value of basis function
    '''
    
    # Initialize the output array v with the same shape as p
    v = np.zeros_like(p)
    
    if etype == 1:
        # Calculate omega^1(x) for each coordinate in p
        for idx, x in enumerate(p):
            if x >= (i-1) * h and x <= i * h:
                v[idx] = (x - (i-1) * h) / h
            else:
                v[idx] = 0
    elif etype == 2:
        # Calculate omega^2(x) for each coordinate in p
        for idx, x in enumerate(p):
            if x >= i * h and x <= (i+1) * h:
                v[idx] = ((i+1) * h - x) / h
            else:
                v[idx] = 0
    else:
        raise ValueError("etype must be 1 or 2")
    
    return v



def gauss_quadrature_points_and_weights():
    """Provides Gauss quadrature points and weights for third order."""
    # Third order Gauss quadrature points and weights on the interval [-1, 1]
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    weights = np.array([5/9, 8/9, 5/9])
    return gauss_points, weights

def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of element
    p: array of arbitrary size 1, 2, or 3, the coordinates
    M: int, the total number of the nodal dofs
    h: int, the element size
    etype: int, basis function type; When type equals to 1, 
    it returns \omega^1(x), when the type equals to 2, it returns the value of function \omega^2(x).
    Outputs
    v: array of size 1, 2, or 3, value of basis function
    '''
    
    # Initialize the output array v with the same shape as p
    v = np.zeros_like(p)
    
    if etype == 1:
        # Calculate omega^1(x) for each coordinate in p
        for idx, x in enumerate(p):
            if x >= (i-1) * h and x <= i * h:
                v[idx] = (x - (i-1) * h) / h
            else:
                v[idx] = 0
    elif etype == 2:
        # Calculate omega^2(x) for each coordinate in p
        for idx, x in enumerate(p):
            if x >= i * h and x <= (i+1) * h:
                v[idx] = ((i+1) * h - x) / h
            else:
                v[idx] = 0
    else:
        raise ValueError("etype must be 1 or 2")
    
    return v

def assemble(M):
    '''Inputs:
    M : number of grid, integer
    Outputs:
    A: mass matrix, 2d array size M*M
    b: right hand side vector, 1d array size M*1
    '''
    
    # Initialize matrix A and vector b
    A = np.zeros((M, M))
    b = np.zeros(M)
    
    # Element size
    h = 1.0 / (M - 1)
    
    # Get Gauss quadrature points and weights
    gauss_points, weights = gauss_quadrature_points_and_weights()
    
    for i in range(M):
        # Local assembly for each element
        for gp, w in zip(gauss_points, weights):
            # Transform from reference to actual element
            x = (i + 0.5 + 0.5 * gp) * h
            
            # Compute basis functions and their derivatives at x
            phi1 = basis(i, np.array([x]), M, h, 1)[0]
            phi2 = basis(i, np.array([x]), M, h, 2)[0]
            
            # SUPG stabilization term (example SUPG stabilization, adjust as needed)
            supg_stabilization = phi1 * phi2
            
            # Assemble mass matrix A
            A[i, i] += w * phi1 * phi1 + supg_stabilization
            if i < M - 1:
                A[i, i + 1] += w * phi1 * phi2
                A[i + 1, i] += w * phi2 * phi1
                A[i + 1, i + 1] += w * phi2 * phi2 + supg_stabilization
            
            # Assemble right hand side vector b (example f(x) = 1)
            f_val = 1  # Assuming f(x) = 1 for simplicity
            b[i] += w * f_val * phi1
            
            if i < M - 1:
                b[i + 1] += w * f_val * phi2
    
    return A, b



def stabilization(A, b):
    '''Inputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    Outputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector 1d array of any size, 1d array of shape (M,)
    '''
    # Parameters
    s_kappa = 1
    a = 200
    C = 50
    M = A.shape[0]
    h = 1.0 / (M - 1)
    
    # Loop through each element and apply Nitsche term and SUPG stabilization
    for i in range(M - 1):
        # Calculate element Peclet number
        P_e = abs(a) * h / (2 * s_kappa)
        
        # Calculate tau
        tau = h / (2 * abs(a)) * (1 / np.tanh(P_e) - 1 / P_e)
        
        # V_kappa
        V_kappa = C * h**-1 * (1 + abs(s_kappa))
        
        # Adjust mass matrix A and vector b with terms
        A[i, i] += V_kappa * tau
        A[i, i + 1] -= V_kappa * tau
        A[i + 1, i] -= V_kappa * tau
        A[i + 1, i + 1] += V_kappa * tau
        
        # Modify b to include Nitsche and stabilization terms (assuming g_D = 0 for simplicity)
        b[i] += V_kappa * tau
        b[i + 1] -= V_kappa * tau
    
    return A, b


try:
    targets = process_hdf5_to_tuple('54.3', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    A = np.array([[ 200.5,   -0.5],[-200.5,    0.5]])
    b = np.array([[-0.99], [ 4.99]])
    assert cmp_tuple_or_list(stabilization(A,b), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    A = np.array([[ 3., 5., 17.],[2., 3., 4.],[1., 2., 3.]])
    b = np.array([[1.], [10.], [3.5]])
    assert cmp_tuple_or_list(stabilization(A,b), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    A = np.array([[ 201.5,   -1.5,    0. ],
           [-201.5,  203. ,   -1.5],
           [   0. , -201.5,    1.5]])
    b = np.array([[-0.12375],
           [ 0.2575 ],
           [ 3.86625]])
    assert cmp_tuple_or_list(stabilization(A,b), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e