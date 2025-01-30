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




def gauss_quad_points_weights(order):
    """
    Returns the quadrature points and weights for Gauss quadrature of the specified order.
    For third order Gauss quadrature, we return 3 points and their respective weights.
    """
    # Third-order Gauss quadrature points and weights
    if order == 3:
        points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
    else:
        raise ValueError("Unsupported Gauss quadrature order")
    
    return points, weights

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

def assemble(M):
    '''Inputs:
    M : number of grid, integer
    Outputs:
    A: mass matrix, 2d array size M*M
    b: right hand side vector , 1d array size M*1
    '''
    # Initialize the mass matrix A and right hand side vector b
    A = np.zeros((M, M))
    b = np.zeros(M)
    
    # Element size
    h = 1.0 / (M - 1)
    
    # Gauss quadrature points and weights for third order
    gauss_points, gauss_weights = gauss_quad_points_weights(3)
    
    # SUPG stabilization parameter, assuming a simple constant for demonstration
    supg_param = 0.01
    
    # Loop over each element and assemble the local contributions to A and b
    for i in range(M - 1):
        # Local stiffness and mass matrices
        A_local = np.zeros((2, 2))
        b_local = np.zeros(2)
        
        # Map quadrature points to element domain
        x_local = 0.5 * h * (gauss_points + 1) + i * h
        
        # Evaluate basis functions at quadrature points
        phi1 = basis(i, x_local, M, h, 1)
        phi2 = basis(i, x_local, M, h, 2)
        
        # Loop over quadrature points
        for j in range(len(gauss_points)):
            # Compute contributions to A_local and b_local
            A_local[0, 0] += gauss_weights[j] * (phi1[j] * phi1[j] + supg_param * phi1[j] * phi1[j]) * h / 2
            A_local[0, 1] += gauss_weights[j] * (phi1[j] * phi2[j] + supg_param * phi1[j] * phi2[j]) * h / 2
            A_local[1, 0] += gauss_weights[j] * (phi2[j] * phi1[j] + supg_param * phi2[j] * phi1[j]) * h / 2
            A_local[1, 1] += gauss_weights[j] * (phi2[j] * phi2[j] + supg_param * phi2[j] * phi2[j]) * h / 2
            
            # Assume a source term f(x) = 1 for the right-hand side
            f_j = 1
            b_local[0] += gauss_weights[j] * phi1[j] * f_j * h / 2
            b_local[1] += gauss_weights[j] * phi2[j] * f_j * h / 2
        
        # Assemble local matrices into global matrix A and vector b
        A[i:i+2, i:i+2] += A_local
        b[i:i+2] += b_local
    
    return A, b


try:
    targets = process_hdf5_to_tuple('54.2', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    M = 11
    assert cmp_tuple_or_list(assemble(M), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    M = 23
    assert cmp_tuple_or_list(assemble(M), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    M = 35
    assert cmp_tuple_or_list(assemble(M), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e