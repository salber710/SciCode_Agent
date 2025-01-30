from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: B-spline basis functions are a family of piecewise-defined polynomials that are used in numerical analysis and computer graphics for curve fitting and surface modeling. The B-spline basis functions are defined recursively using the Cox-de Boor recursion formula. The recursion starts with piecewise constant functions (degree 0) and builds up to higher degrees. The recursion formula for B-splines is:
# 
# N_{i,0}(x) = 1 if Xi[i] <= x < Xi[i+1], else 0
# N_{i,p}(x) = ((x - Xi[i]) / (Xi[i+p] - Xi[i])) * N_{i,p-1}(x) + ((Xi[i+p+1] - x) / (Xi[i+p+1] - Xi[i+1])) * N_{i+1,p-1}(x)
# 
# where N_{i,p}(x) is the B-spline basis function of degree p, Xi is the knot vector, and i is the index of the basis function. The function Bspline(xi, i, p, Xi) computes the value of the B-spline basis function of degree p at the knot index xi.


def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index , integer
    p : polynomial degree of basis function , integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2 or 3
    '''
    if p == 0:
        # Base case: degree 0 B-spline
        return np.array([1.0 if Xi[i] <= xi < Xi[i+1] else 0.0])
    else:
        # Recursive case: degree p B-spline
        # Calculate the coefficients alpha and beta
        if Xi[i+p] != Xi[i]:
            alpha = (xi - Xi[i]) / (Xi[i+p] - Xi[i])
        else:
            alpha = 0.0
        
        if Xi[i+p+1] != Xi[i+1]:
            beta = (Xi[i+p+1] - xi) / (Xi[i+p+1] - Xi[i+1])
        else:
            beta = 0.0
        
        # Recursive calculation
        left = Bspline(xi, i, p-1, Xi)
        right = Bspline(xi, i+1, p-1, Xi)
        
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