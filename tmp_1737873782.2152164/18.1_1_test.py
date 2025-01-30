from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def Bspline(x, i, p, Xi):
    '''Inputs:
    x : evaluation point, float
    i : polynomial index, integer
    p : polynomial degree of basis function, integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    The value of the B-spline basis function N_{i,p}(x)
    '''
    if p == 0:
        # Base case: piecewise constant basis function
        if Xi[i] <= x < Xi[i + 1]:
            return 1.0
        else:
            return 0.0
    else:
        # Recursive definition for p > 0
        # Calculate coefficients with division by zero handling
        if Xi[i + p] - Xi[i] != 0:
            alpha = (x - Xi[i]) / (Xi[i + p] - Xi[i])
        else:
            alpha = 0.0

        if Xi[i + p + 1] - Xi[i + 1] != 0:
            beta = (Xi[i + p + 1] - x) / (Xi[i + p + 1] - Xi[i + 1])
        else:
            beta = 0.0

        # Recursive calls
        return alpha * Bspline(x, i, p - 1, Xi) + beta * Bspline(x, i + 1, p - 1, Xi)


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