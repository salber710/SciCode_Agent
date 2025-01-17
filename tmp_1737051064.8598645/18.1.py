import numpy as np



# Background: B-spline basis functions are a family of piecewise-defined polynomials that are used in numerical analysis and computer graphics for curve fitting and surface modeling. 
# They are defined recursively using the Cox-de Boor recursion formula. The recursion starts with piecewise constant functions (degree 0) and builds up to higher degrees.
# The recursion formula for B-splines is:
# N_{i,0}(x) = 1 if Xi[i] <= x < Xi[i+1], else 0
# N_{i,p}(x) = (x - Xi[i]) / (Xi[i+p] - Xi[i]) * N_{i,p-1}(x) + (Xi[i+p+1] - x) / (Xi[i+p+1] - Xi[i+1]) * N_{i+1,p-1}(x)
# where N_{i,p}(x) is the B-spline basis function of degree p, defined over the knot vector Xi.


def Bspline(x, i, p, Xi):
    '''Inputs:
    x : evaluation point, float
    i : polynomial index, integer
    p : polynomial degree of basis function, integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    float : value of the B-spline basis function at x
    '''
    if p == 0:
        # Base case: degree 0
        if Xi[i] <= x < Xi[i+1]:
            return 1.0
        else:
            return 0.0
    else:
        # Recursive case: degree p
        # Calculate the coefficients for the recursive formula
        if Xi[i+p] != Xi[i]:
            alpha = (x - Xi[i]) / (Xi[i+p] - Xi[i])
        else:
            alpha = 0.0
        
        if Xi[i+p+1] != Xi[i+1]:
            beta = (Xi[i+p+1] - x) / (Xi[i+p+1] - Xi[i+1])
        else:
            beta = 0.0
        
        # Recursive calculation
        return alpha * Bspline(x, i, p-1, Xi) + beta * Bspline(x, i+1, p-1, Xi)


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
