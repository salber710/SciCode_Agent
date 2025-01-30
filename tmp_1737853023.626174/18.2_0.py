import numpy as np

# Background: B-spline basis functions are a family of piecewise-defined polynomials that are used in numerical analysis and computer graphics for curve fitting and modeling. The B-spline basis functions are defined recursively using the Cox-de Boor recursion formula. The recursion starts with piecewise constant functions (degree 0) and builds up to higher degrees. The recursion formula for B-splines is:
# 
# N_{i,0}(x) = 1 if Xi[i] <= x < Xi[i+1], else 0
# N_{i,p}(x) = ((x - Xi[i]) / (Xi[i+p] - Xi[i])) * N_{i,p-1}(x) + ((Xi[i+p+1] - x) / (Xi[i+p+1] - Xi[i+1])) * N_{i+1,p-1}(x)
# 
# where N_{i,p}(x) is the B-spline basis function of degree p, Xi is the knot vector, and i is the index of the basis function. The function Bspline(xi, i, p, Xi) is designed to evaluate these basis functions at a given knot index xi, for a given polynomial index i and degree p, using the knot vector Xi.


def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, float
    i : polynomial index, integer
    p : polynomial degree of basis function, integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2 or 3
    '''
    if not isinstance(i, int) or not isinstance(p, int):
        raise TypeError("Indices i and degree p must be integers.")
    
    if len(Xi) == 0:
        raise ValueError("Knot vector Xi cannot be empty.")
    
    if i < 0 or i >= len(Xi) - p:
        raise IndexError("Index i is out of the valid range.")
    
    if p < 0:
        raise ValueError("Polynomial degree p cannot be negative.")
    
    if p == 0:
        # Base case: degree 0
        return np.array([1.0 if Xi[i] <= xi < Xi[i+1] else 0.0])
    else:
        # Recursive case: degree p
        left_term = 0.0
        right_term = 0.0
        
        if i + p < len(Xi):
            left_denom = Xi[i+p] - Xi[i]
            if left_denom != 0:
                left_term = (xi - Xi[i]) / left_denom * Bspline(xi, i, p-1, Xi)
        
        if i + p + 1 < len(Xi):
            right_denom = Xi[i+p+1] - Xi[i+1]
            if right_denom != 0:
                right_term = (Xi[i+p+1] - xi) / right_denom * Bspline(xi, i+1, p-1, Xi)
        
        return left_term + right_term



# Background: NURBS (Non-Uniform Rational B-Splines) are an extension of B-splines that allow for the representation of both standard analytical shapes (like circles) and free-form curves. NURBS are defined by a set of control points, a knot vector, and a set of weights. The weights allow for the representation of conic sections and other complex shapes. The NURBS basis function is a weighted version of the B-spline basis function. The NURBS basis function R_{i,p}(x) is given by:
# 
# R_{i,p}(x) = (N_{i,p}(x) * w_i) / (sum(N_{j,p}(x) * w_j for all j))
# 
# where N_{i,p}(x) is the B-spline basis function, w_i is the weight associated with the i-th control point, and the denominator is the weighted sum of all B-spline basis functions at x. In a 2D NURBS surface, the basis function is evaluated as a product of two 1D NURBS basis functions.


def NURBS_2D(xi_1, xi_2, i_1, i_2, p_1, p_2, n_1, n_2, Xi_1, Xi_2, w):
    '''Inputs:
    xi_1 : parameter coordinate at the first dof, float
    xi_2 : parameter coordinate at the second dof, float
    i_1 : index of the basis function to be evaluated at the first dof, integer
    i_2 : index of the basis function to be evaluated at the second dof, integer
    p_1 : polynomial degree of the basis function to be evaluated at the first dof, integer
    p_2 : polynomial degree of the basis function to be evaluated at the second dof, integer
    n_1 : total number of basis function at the first dof, integer
    n_2 : total number of basis function at the second dof, integer
    Xi_1 : knot vector of arbitrary size , 1d array
    Xi_2 : knot vector of arbitrary size , 1d array
    w : array storing NURBS weights, 1d array
    Outputs:
    N : value of the basis functions evaluated at the given paramter coordinates, 1d array of size 1 or 2
    '''
    
    # Evaluate the B-spline basis functions for the given indices and degrees
    N_i1_p1 = Bspline(xi_1, i_1, p_1, Xi_1)
    N_i2_p2 = Bspline(xi_2, i_2, p_2, Xi_2)
    
    # Calculate the numerator of the NURBS basis function
    numerator = N_i1_p1 * N_i2_p2 * w[i_1 * n_2 + i_2]
    
    # Calculate the denominator of the NURBS basis function
    denominator = 0.0
    for j_1 in range(n_1):
        for j_2 in range(n_2):
            N_j1_p1 = Bspline(xi_1, j_1, p_1, Xi_1)
            N_j2_p2 = Bspline(xi_2, j_2, p_2, Xi_2)
            denominator += N_j1_p1 * N_j2_p2 * w[j_1 * n_2 + j_2]
    
    # Calculate the NURBS basis function value
    N = numerator / denominator if denominator != 0 else 0.0
    
    return np.array([N])

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('18.2', 3)
target = targets[0]

p_1 = 2
p_2 = 2
Xi_1 = [0, 0, 0, 1, 2, 2, 3, 4, 4, 4]
Xi_2 = [0, 0, 0, 1, 2, 2, 2]
w = [0]
i_1 = 2
i_2 = 1
xi_1 = 1
xi_2 = 0
n_1 = len(Xi_1) - p_1 - 1
n_2 = len(Xi_2) - p_2 - 1
assert np.allclose(NURBS_2D(xi_1, xi_2, i_1, i_2, p_1, p_2, n_1, n_2, Xi_1, Xi_2, w), target)
target = targets[1]

Xi_1 = [0,0,0,1,2,3,4,4,5,5,5]
Xi_2 = [0,0,0,1,2,3,3,3]
xi_1 = 2.5
xi_2 = 1
i_1 = 3
i_2 = 2
p_1 = 2
p_2 = 2
n_1 = len(Xi_1) - p_1 - 1
n_2 = len(Xi_2) - p_2 - 1
w = np.ones(n_1*n_2)
assert np.allclose(NURBS_2D(xi_1, xi_2, i_1, i_2, p_1, p_2, n_1, n_2, Xi_1, Xi_2, w), target)
target = targets[2]

Xi_1 = [0,0,0,0.5,1,1,1]
Xi_2 = [0,0,0,1,1,1]
p_1 = 2
p_2 = 2
n_1 = len(Xi_1) - p_1 - 1
n_2 = len(Xi_2) - p_2 - 1
w = [1,1,1,1,1,1,1,1,1,1,1,1]
xi_1 = 0.2
xi_2 = 0.5
i_1  = 2
i_2 = 1
assert np.allclose(NURBS_2D(xi_1, xi_2, i_1, i_2, p_1, p_2, n_1, n_2, Xi_1, Xi_2, w), target)
