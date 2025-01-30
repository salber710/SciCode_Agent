from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index , integer
    p : polynomial degree of basis function , integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2 or 3
    '''
    # Create a 2D list to store the basis function values
    n = len(Xi) - 1
    N = [[0.0] * (p + 1) for _ in range(n)]
    
    # Base case: p = 0
    for j in range(n):
        N[j][0] = 1.0 if Xi[j] <= xi < Xi[j + 1] else 0.0
    
    # Fill the table iteratively for higher degrees
    for d in range(1, p + 1):
        for j in range(n - d):
            # Compute alpha and beta using the knot vector
            denom1 = Xi[j + d] - Xi[j]
            denom2 = Xi[j + d + 1] - Xi[j + 1]

            alpha = (xi - Xi[j]) / denom1 if denom1 != 0 else 0.0
            beta = (Xi[j + d + 1] - xi) / denom2 if denom2 != 0 else 0.0
            
            # Calculate the basis function value using previous values
            N[j][d] = alpha * N[j][d - 1] + beta * N[j + 1][d - 1]

    # Return the basis function value at the specified degree and index
    return N[i][p]


try:
    targets = process_hdf5_to_tuple('18.1', 3)
    target = targets[0]
    xi =0.1
    i = 1
    p = 2
    Xi = [0,0,0,1,1,1]
    assert np.allclose(Bspline(xi, i, p, Xi), target)

    target = targets[1]
    xi = 1.5
    i = 1
    p = 3
    Xi = [0,0,0,1,1,1,2,2,2]
    assert np.allclose(Bspline(xi, i, p, Xi), target)

    target = targets[2]
    xi = 0.5
    i = 1
    p = 3
    Xi = [0,0,1,1,2,2,3,3]
    assert np.allclose(Bspline(xi, i, p, Xi), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e