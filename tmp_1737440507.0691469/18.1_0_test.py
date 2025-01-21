import numpy as np



# Background: B-splines (Basis splines) are a family of piecewise-defined polynomials
# used in computational graphics to approximate curves and surfaces. They are defined 
# using a set of control points and a knot vector, which determines where and how the 
# control points affect the B-spline curve. The degree of the B-spline polynomial (p) 
# controls the smoothness and continuity of the curve or surface. B-spline basis functions 
# are recursively defined, and this property allows the B-spline to have local support 
# and influence only within a certain range of the knot vector. The recursive definition 
# is given by the Cox-de Boor recursion formula, which calculates the value of a B-spline 
# basis function at a given parameter value. The recursion starts with piecewise constant 
# functions (degree 0) and builds up to higher degrees.

def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index , integer
    p : polynomial degree of basis function , integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2 or 3
    '''

    
    # Base case for recursion: when p == 0
    if p == 0:
        if Xi[i] <= xi < Xi[i+1]:
            return 1.0
        else:
            return 0.0

    # Recursive definition using Cox-de Boor formula
    # Compute coefficients alpha and beta for the recursive relation
    alpha = 0.0
    beta = 0.0
    
    # Avoid division by zero
    if Xi[i+p] != Xi[i]:
        alpha = (xi - Xi[i]) / (Xi[i+p] - Xi[i])
    if Xi[i+p+1] != Xi[i+1]:
        beta = (Xi[i+p+1] - xi) / (Xi[i+p+1] - Xi[i+1])
    
    # Recursively compute the value of the basis function
    return alpha * Bspline(xi, i, p-1, Xi) + beta * Bspline(xi, i+1, p-1, Xi)

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
