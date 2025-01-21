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



# Background: Non-Uniform Rational B-Splines (NURBS) are a mathematical model often used in computer graphics for generating and representing curves and surfaces.
# NURBS extend B-splines by incorporating weights, which provide more control over the shape of the curve. A NURBS curve is defined by a set of control points,
# a knot vector, and a set of weights. The weights allow NURBS to represent exact conic sections and freeform curves. The NURBS basis functions are defined as a
# weighted sum of B-spline basis functions, where each B-spline function is scaled by its corresponding weight. The evaluation of a NURBS basis function involves
# calculating the B-spline basis functions, and then computing the weighted sum divided by the sum of all weighted B-spline basis functions to normalize.


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
    
    # Calculate the B-spline basis function for each parameter direction
    N_1 = np.zeros(n_1)
    N_2 = np.zeros(n_2)
    
    for j in range(n_1):
        N_1[j] = Bspline(xi_1, j, p_1, Xi_1)
    
    for k in range(n_2):
        N_2[k] = Bspline(xi_2, k, p_2, Xi_2)
    
    # Calculate the weighted B-spline basis functions
    R = np.zeros((n_1, n_2))
    denominator = 0.0  # This will store the sum of all w[j, k] * N_1[j] * N_2[k]
    
    for j in range(n_1):
        for k in range(n_2):
            R[j, k] = w[j * n_2 + k] * N_1[j] * N_2[k]
            denominator += R[j, k]
    
    # Normalize the basis function value with the denominator
    if denominator != 0:
        N = R[i_1, i_2] / denominator
    else:
        N = 0.0
    
    return N

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
