import numpy as np



# Background: B-spline basis functions are a family of piecewise-defined polynomials used in numerical analysis 
# and computer graphics for curve and surface representation. They are defined recursively using the Cox-de Boor 
# recursion formula. The degree of the B-spline (p) determines the polynomial degree of the segments. The knot 
# vector (Xi) specifies where and how the control points affect the B-spline curve. For a given knot index (xi) 
# and polynomial index (i), the B-spline function is defined as follows:
# - If the polynomial degree p is 0, the basis function is 1 if the variable x falls within the knot span, 
#   otherwise, it is 0.
# - For p > 0, the basis function is a linear combination of two basis functions of degree p-1, weighted by 
#   coefficients alpha and beta, which are determined by the position of x relative to the knots.
# This recursive definition allows constructing B-spline basis functions of any degree.


def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index, integer
    p : polynomial degree of basis function, integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2, or 3
    '''
    if p == 0:
        # Base case: Check if the xi is within the knot span defined by Xi[i] and Xi[i+1]
        return np.array([1.0 if Xi[i] <= xi < Xi[i + 1] else 0.0])
    else:
        # Recursive case: Compute the B-spline basis function for degree p using the recursive formula
        denom1 = Xi[i + p] - Xi[i]
        denom2 = Xi[i + p + 1] - Xi[i + 1]
        
        # Calculate alpha and beta coefficients safely to avoid division by zero
        alpha = 0.0 if denom1 == 0 else (xi - Xi[i]) / denom1
        beta = 0.0 if denom2 == 0 else (Xi[i + p + 1] - xi) / denom2
        
        # Recursively calculate the B-spline basis function
        left = Bspline(xi, i, p - 1, Xi)
        right = Bspline(xi, i + 1, p - 1, Xi)
        
        # Combine left and right basis functions with alpha and beta coefficients
        return alpha * left + beta * right

from scicode.parse.parse import process_hdf5_to_tuple
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
