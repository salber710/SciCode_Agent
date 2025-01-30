from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index, integer
    p : polynomial degree of basis function, integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    Scalar value representing the B-spline basis function value at xi.
    '''
    
    if p == 0:
        # Base case: if degree is 0, use the piecewise constant function definition
        if Xi[i] <= xi < Xi[i+1]:
            return 1.0
        else:
            return 0.0
        
    else:
        # Recursive case: calculate the B-spline basis function using the Cox-de Boor recursion formula
        
        # Avoid division by zero by checking the denominator
        denom1 = Xi[i+p] - Xi[i]
        denom2 = Xi[i+p+1] - Xi[i+1]
        
        # Calculate the coefficients alpha and beta
        alpha = 0.0 if denom1 == 0 else (xi - Xi[i]) / denom1 * Bspline(xi, i, p-1, Xi)
        beta = 0.0 if denom2 == 0 else (Xi[i+p+1] - xi) / denom2 * Bspline(xi, i+1, p-1, Xi)
        
        # Return the weighted sum of the recursive calculations
        return alpha + beta




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
    
    def Bspline(xi, i, p, Xi):
        '''Evaluate the B-spline basis function value at xi.'''
        if p == 0:
            if Xi[i] <= xi < Xi[i+1]:
                return 1.0
            else:
                return 0.0
        else:
            denom1 = Xi[i+p] - Xi[i]
            denom2 = Xi[i+p+1] - Xi[i+1]
            
            alpha = 0.0 if denom1 == 0 else (xi - Xi[i]) / denom1 * Bspline(xi, i, p-1, Xi)
            beta = 0.0 if denom2 == 0 else (Xi[i+p+1] - xi) / denom2 * Bspline(xi, i+1, p-1, Xi)
            
            return alpha + beta

    # Initialize the numerator and denominator for the NURBS basis function
    numerator = 0.0
    denominator = 0.0
    
    # Loop over all combinations of basis functions
    for i in range(n_1):
        for j in range(n_2):
            # Evaluate the B-spline basis function for both xi_1 and xi_2
            N_i_1 = Bspline(xi_1, i, p_1, Xi_1)
            N_j_2 = Bspline(xi_2, j, p_2, Xi_2)
            
            # Calculate the product of the basis function and weight
            product = N_i_1 * N_j_2 * w[i * n_2 + j]
            
            # Add to the numerator and denominator
            numerator += product * N_i_1 * N_j_2
            denominator += product
    
    # Compute the NURBS basis function value
    N = numerator / denominator if denominator != 0 else 0.0
    
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