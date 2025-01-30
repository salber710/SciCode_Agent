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



def assemble(M):
    '''Inputs:
    M : number of grid, integer
    Outputs:
    A: mass matrix, 2d array size M*M
    b: right hand side vector , 1d array size M*1
    '''
    # Initialize the mass matrix A and right-hand side vector b
    A = np.zeros((M, M))
    b = np.zeros(M)

    # Define the grid spacing h (assuming uniform grid over the interval [0, 1])
    h = 1.0 / (M + 1)

    # Define the quadrature points and weights for third order Gaussian quadrature
    # These are standard for the reference interval [-1, 1]
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])

    # Transform the Gauss points to the interval [0, h] for each element
    transformed_points = (gauss_points + 1) * h / 2
    transformed_weights = gauss_weights * h / 2

    # Loop over all elements to construct A and b
    for i in range(1, M+1):
        # Calculate the contributions from each quadrature point
        for gp, gw in zip(transformed_points, transformed_weights):
            # Basis function values at the quadrature point
            phi_1 = (gp - (i-1) * h) / h  # \omega^1(x)
            phi_2 = ((i+1) * h - gp) / h  # \omega^2(x)

            # SUPG stabilization term can be added here if needed

            # Assemble the local contributions into A and b
            A[i-1, i-1] += gw * phi_1 * phi_1
            A[i-1, i] += gw * phi_1 * phi_2
            A[i, i-1] += gw * phi_2 * phi_1
            A[i, i] += gw * phi_2 * phi_2

            # For simplicity, let's assume a constant source term for the right-hand side vector
            b[i-1] += gw * phi_1
            b[i] += gw * phi_2

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