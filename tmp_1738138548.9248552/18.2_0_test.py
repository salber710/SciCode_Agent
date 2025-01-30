from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def Bspline(xi, i, p, Xi):
    x = Xi[xi]
    if p == 0:
        return np.array([1.0 if Xi[i] <= x < Xi[i+1] else 0.0])
    else:
        alpha = (x - Xi[i]) / (Xi[i+p] - Xi[i]) if Xi[i+p] != Xi[i] else 0
        beta = (Xi[i+p+1] - x) / (Xi[i+p+1] - Xi[i+1]) if Xi[i+p+1] != Xi[i+1] else 0
        return np.array([alpha * Bspline(xi, i, p-1, Xi)[0] + beta * Bspline(xi, i+1, p-1, Xi)[0]])



# Background: 
# NURBS (Non-Uniform Rational B-Splines) are an extension of B-splines that allow for the representation of complex shapes such as circles and ellipses. 
# They are defined by a set of control points, a knot vector, and a set of weights. The weights allow for the rational aspect of NURBS, enabling the representation of conic sections.
# The NURBS basis function is computed as the weighted sum of B-spline basis functions, divided by the sum of all weighted B-spline basis functions (the denominator).
# The B-spline basis functions are evaluated using the Cox-de Boor recursion formula, which is already implemented in the Bspline function.
# The NURBS basis function in 2D is a product of two 1D NURBS basis functions, one for each parameter direction.


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
    
    # Evaluate the B-spline basis functions for the given parameters
    N_i1_p1 = Bspline(xi_1, i_1, p_1, Xi_1)[0]
    N_i2_p2 = Bspline(xi_2, i_2, p_2, Xi_2)[0]
    
    # Calculate the numerator of the NURBS basis function
    R_i1_i2 = N_i1_p1 * N_i2_p2 * w[i_1 * n_2 + i_2]
    
    # Calculate the denominator (sum of all weighted B-spline basis functions)
    denominator = 0.0
    for j_1 in range(n_1):
        for j_2 in range(n_2):
            N_j1_p1 = Bspline(xi_1, j_1, p_1, Xi_1)[0]
            N_j2_p2 = Bspline(xi_2, j_2, p_2, Xi_2)[0]
            denominator += N_j1_p1 * N_j2_p2 * w[j_1 * n_2 + j_2]
    
    # Calculate the NURBS basis function value
    N = R_i1_i2 / denominator if denominator != 0 else 0.0
    
    return np.array([N])


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e