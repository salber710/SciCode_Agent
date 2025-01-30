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
    1d array of size 1，2 or 3
    '''

    # Base case for B-spline of degree 0
    if p == 0:
        if Xi[i] <= xi < Xi[i+1]:
            return 1.0
        else:
            return 0.0

    # Calculate alpha term
    if Xi[i+p] != Xi[i]:
        alpha = (xi - Xi[i]) / (Xi[i+p] - Xi[i])
    else:
        alpha = 0.0

    # Calculate beta term
    if Xi[i+p+1] != Xi[i+1]:
        beta = (Xi[i+p+1] - xi) / (Xi[i+p+1] - Xi[i+1])
    else:
        beta = 0.0

    # Recursive computation of B-spline basis function
    return alpha * Bspline(xi, i, p-1, Xi) + beta * Bspline(xi, i+1, p-1, Xi)


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