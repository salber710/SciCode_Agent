from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def basis(i, p, M, h, etype):

    v = np.zeros_like(p)
    if etype == 1:
        x_i_minus_1 = (i - 1) * h
        x_i = i * h
        v = np.piecewise(p, [p < x_i_minus_1, (p >= x_i_minus_1) & (p <= x_i), p > x_i], [0, lambda x: (x - x_i_minus_1) / h, 0])
    elif etype == 2:
        x_i = i * h
        x_i_plus_1 = (i + 1) * h
        v = np.piecewise(p, [p < x_i, (p >= x_i) & (p <= x_i_plus_1), p > x_i_plus_1], [0, lambda x: (x_i_plus_1 - x) / h, 0])
    return v



def assemble(M):
    A = np.zeros((M, M))
    b = np.zeros(M)
    
    # Third-order Gaussian quadrature points and weights
    points = [-np.sqrt(3/5), 0, np.sqrt(3/5)]
    weights = [5/9, 8/9, 5/9]
    
    # Element size
    h = 1 / (M - 1)
    
    # SUPG parameter
    tau = h / 2
    
    # Loop through each element
    for i in range(M - 1):
        x_left = i * h
        x_right = (i + 1) * h
        
        # Local matrices and vectors
        A_local = np.zeros((2, 2))
        b_local = np.zeros(2)
        
        # Integration using Gaussian quadrature
        for point, weight in zip(points, weights):
            # Map quadrature points to the actual element
            xi = x_left + (x_right - x_left) * (point + 1) / 2
            
            # Quadratic shape functions
            N1 = 2 * ((xi - x_left) / h) * ((xi - x_right) / h)
            N2 = 4 * ((xi - x_left) / h) * ((xi - x_right) / h)
            dN1_dx = (4 * xi - 2 * (x_left + x_right)) / h**2
            dN2_dx = (8 * xi - 4 * (x_left + x_right)) / h**2
            
            # Mass matrix contribution
            A_local[0, 0] += weight * N1 * N1 * h
            A_local[0, 1] += weight * N1 * N2 * h
            A_local[1, 0] += weight * N2 * N1 * h
            A_local[1, 1] += weight * N2 * N2 * h
            
            # Right-hand side vector contribution (assuming f(xi) = 1)
            b_local[0] += weight * N1 * h
            b_local[1] += weight * N2 * h
            
            # SUPG stabilization terms
            A_local[0, 0] += tau * dN1_dx * dN1_dx * weight * h
            A_local[0, 1] += tau * dN1_dx * dN2_dx * weight * h
            A_local[1, 0] += tau * dN2_dx * dN1_dx * weight * h
            A_local[1, 1] += tau * dN2_dx * dN2_dx * weight * h
        
        # Assemble local contributions into global matrices
        A[i:i+2, i:i+2] += A_local
        b[i:i+2] += b_local
    
    return A, b



# Background: 
# Nitsche's method is a technique used to impose boundary conditions weakly in finite element methods. 
# It involves adding terms to the variational formulation that penalize the difference between the 
# numerical solution and the boundary condition. The SUPG (Streamline Upwind Petrov-Galerkin) method 
# is a stabilization technique used to handle convection-dominated problems by adding a stabilization 
# term to the weak form of the equation. The parameters involved are:
# - s_kappa = 1, a stabilization parameter.
# - a = 200, a convection coefficient.
# - V_kappa = C * h^(-1) * (1 + |s_kappa|), where C = 50, is a penalty parameter.
# - tau is a stabilization parameter calculated using the element Peclet number P^e = |a|h/(2*kappa).
#   The formula for tau is tau = h/(2|a|) * (coth(P^e) - 1/P^e), where coth is the hyperbolic cotangent.


def stabilization(A, b):
    '''Inputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    Outputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector 1d array of any size, 1d array of shape (M,)
    '''
    M = A.shape[0]
    h = 1 / (M - 1)
    s_kappa = 1
    a = 200
    C = 50
    kappa = 1  # Assuming kappa is 1 for simplicity, adjust as needed.
    
    # Calculate V_kappa
    V_kappa = C * h**(-1) * (1 + abs(s_kappa))
    
    # Calculate tau using the element Peclet number
    P_e = abs(a) * h / (2 * kappa)
    tau = h / (2 * abs(a)) * (np.cosh(P_e) / np.sinh(P_e) - 1 / P_e)
    
    # Adjust the mass matrix A and right hand side vector b
    for i in range(M - 1):
        # Nitsche term contribution
        A[i, i] += V_kappa
        A[i + 1, i + 1] += V_kappa
        A[i, i + 1] -= V_kappa
        A[i + 1, i] -= V_kappa
        
        # SUPG stabilization term contribution
        A[i, i] += tau
        A[i + 1, i + 1] += tau
        
        # Adjust the right hand side vector b
        b[i] += V_kappa
        b[i + 1] += V_kappa
    
    return A, b


try:
    targets = process_hdf5_to_tuple('54.3', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    A = np.array([[ 200.5,   -0.5],[-200.5,    0.5]])
    b = np.array([[-0.99], [ 4.99]])
    assert cmp_tuple_or_list(stabilization(A,b), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    A = np.array([[ 3., 5., 17.],[2., 3., 4.],[1., 2., 3.]])
    b = np.array([[1.], [10.], [3.5]])
    assert cmp_tuple_or_list(stabilization(A,b), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    A = np.array([[ 201.5,   -1.5,    0. ],
           [-201.5,  203. ,   -1.5],
           [   0. , -201.5,    1.5]])
    b = np.array([[-0.12375],
           [ 0.2575 ],
           [ 3.86625]])
    assert cmp_tuple_or_list(stabilization(A,b), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e