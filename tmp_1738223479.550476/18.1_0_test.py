from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: B-splines (Basis splines) are piecewise-defined polynomials used in numerical analysis for curve fitting and data approximation.
# They are defined by their degree (p) and a knot vector (Xi), which determines where and how the polynomials join.
# The B-spline basis functions are defined recursively using the Cox-de Boor recursion formula.
# The base case is the piecewise constant functions, and the recursion extends to higher degree polynomials.
# To evaluate a B-spline basis function at a given knot index (xi), polynomial index (i), and degree (p),
# we must consider the continuity and differentiability at the knots prescribed by Xi.
# The recursion relation involves linear combinations of lower-degree basis functions weighted by coefficients
# that depend on the knot vector Xi.


def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index , integer
    p : polynomial degree of basis function , integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2 or 3
    '''
    # Base case: p = 0
    if p == 0:
        # Return 1 if xi is within the range of the i-th knot interval, otherwise 0
        return 1.0 if Xi[i] <= xi < Xi[i + 1] else 0.0

    # Recursive case: p > 0
    # Calculate the coefficients alpha and beta using the knot vector
    if Xi[i + p] != Xi[i]:
        alpha = (xi - Xi[i]) / (Xi[i + p] - Xi[i])
    else:
        alpha = 0.0

    if Xi[i + p + 1] != Xi[i + 1]:
        beta = (Xi[i + p + 1] - xi) / (Xi[i + p + 1] - Xi[i + 1])
    else:
        beta = 0.0

    # Recursive evaluation of the basis functions
    left = Bspline(xi, i, p - 1, Xi)
    right = Bspline(xi, i + 1, p - 1, Xi)

    # Return the linear combination of the lower-order basis functions
    return alpha * left + beta * right


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