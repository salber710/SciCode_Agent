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
    # Bottom-up dynamic programming approach with tabulation
    n = len(Xi) - 1
    dp = [[0.0 for _ in range(p + 1)] for _ in range(n)]

    # Initialize the base case: p = 0
    for j in range(n):
        dp[j][0] = 1.0 if Xi[j] <= xi < Xi[j + 1] else 0.0

    # Build up the table for higher degrees
    for degree in range(1, p + 1):
        for j in range(n - degree):
            # Calculate alpha and beta
            denom1 = Xi[j + degree] - Xi[j]
            denom2 = Xi[j + degree + 1] - Xi[j + 1]

            alpha = (xi - Xi[j]) / denom1 if denom1 != 0 else 0.0
            beta = (Xi[j + degree + 1] - xi) / denom2 if denom2 != 0 else 0.0

            # Fill the table using previously computed values
            dp[j][degree] = alpha * dp[j][degree - 1] + beta * dp[j + 1][degree - 1]

    # Return the desired basis function value
    return dp[i][p]


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