import numpy as np

# Background: 
# In the context of finite element analysis, shape functions are used to interpolate the solution over an element. 
# For a 1D linear element, the shape functions are linear polynomials that depend on the position within the element.
# The function ω^1(x) is defined for the interval [x_{i-1}, x_i] and linearly interpolates between these two points.
# Similarly, ω^2(x) is defined for the interval [x_i, x_{i+1}] and interpolates between these points.
# The parameter 'h' represents the element size, which is the distance between consecutive nodes.
# The function 'basis' computes the value of these shape functions based on the type specified by 'etype'.

def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of element
    p: array of arbitrary size 1,2, or 3, the coordinates
    M: int, the total number of the nodal dofs
    h: int, the element size
    etype: int, basis function type; When type equals to 1, 
    it returns ω^1(x), when the type equals to 2, it returns the value of function ω^2(x).
    Outputs
    v: array of size 1,2, or 3, value of basis function
    '''

    
    v = np.zeros_like(p)
    
    if etype == 1:
        # Calculate ω^1(x) = (x - x_{i-1}) / h_{i-1} for x_{i-1} <= x <= x_i
        for j in range(len(p)):
            if i > 0 and p[j] >= (i-1)*h and p[j] <= i*h:
                v[j] = (p[j] - (i-1)*h) / h
            else:
                v[j] = 0.0
    elif etype == 2:
        # Calculate ω^2(x) = (x_{i+1} - x) / h_i for x_i <= x <= x_{i+1}
        for j in range(len(p)):
            if i < M-1 and p[j] >= i*h and p[j] <= (i+1)*h:
                v[j] = ((i+1)*h - p[j]) / h
            else:
                v[j] = 0.0
    else:
        raise ValueError("etype must be 1 or 2")
    
    return v


# Background: 
# In finite element analysis, the mass matrix and the right-hand side vector are crucial components for solving differential equations numerically.
# The mass matrix A is typically assembled by integrating the product of shape functions over each element. 
# The right-hand side vector b is assembled by integrating the source term (or load) against the shape functions.
# Third-order Gaussian quadrature is a numerical integration method that provides accurate results by evaluating the integrand at specific points (Gauss points) and weights.
# SUPG (Streamline Upwind Petrov-Galerkin) is a stabilization technique used to improve the numerical solution of advection-dominated problems by adding a stabilization term to the weak form.
# The function 'assemble' will construct the mass matrix A and the right-hand side vector b using these techniques.


def assemble(M):
    '''Inputs:
    M : number of grid, integer
    Outputs:
    A: mass matrix, 2d array size M*M
    b: right hand side vector , 1d array size M*1
    '''
    
    # Initialize the mass matrix A and the right-hand side vector b
    A = np.zeros((M, M))
    b = np.zeros(M)
    
    # Define the Gauss points and weights for third-order Gaussian quadrature
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    # Element size
    h = 1.0 / (M - 1)  # Assuming a uniform grid over the domain [0, 1]
    
    # Loop over each element
    for i in range(M - 1):
        # Local element matrix and vector
        A_local = np.zeros((2, 2))
        b_local = np.zeros(2)
        
        # Loop over Gauss points
        for gp, gw in zip(gauss_points, gauss_weights):
            # Map Gauss point to the element
            xi = 0.5 * (1 + gp)  # Transform from [-1, 1] to [0, 1]
            x = i * h + xi * h  # Physical coordinate
            
            # Evaluate shape functions and their derivatives at the Gauss point
            N1 = 1 - xi
            N2 = xi
            dN1_dx = -1 / h
            dN2_dx = 1 / h
            
            # SUPG stabilization term (simplified for demonstration)
            tau = h / 2  # Stabilization parameter
            
            # Assemble local mass matrix
            A_local[0, 0] += gw * N1 * N1 * h
            A_local[0, 1] += gw * N1 * N2 * h
            A_local[1, 0] += gw * N2 * N1 * h
            A_local[1, 1] += gw * N2 * N2 * h
            
            # Assemble local right-hand side vector
            # Assuming a source term f(x) = 1 for simplicity
            f = 1
            b_local[0] += gw * N1 * f * h
            b_local[1] += gw * N2 * f * h
            
            # Add SUPG stabilization to the local matrix
            A_local[0, 0] += tau * gw * dN1_dx * dN1_dx * h
            A_local[0, 1] += tau * gw * dN1_dx * dN2_dx * h
            A_local[1, 0] += tau * gw * dN2_dx * dN1_dx * h
            A_local[1, 1] += tau * gw * dN2_dx * dN2_dx * h
        
        # Assemble global mass matrix and right-hand side vector
        A[i:i+2, i:i+2] += A_local
        b[i:i+2] += b_local
    
    return A, b



# Background: 
# In finite element analysis, stabilization techniques such as SUPG (Streamline Upwind Petrov-Galerkin) and Nitsche's method are used to improve the numerical solution of differential equations, especially in advection-dominated problems.
# The SUPG stabilization term is added to the weak form to counteract numerical instabilities by modifying the test functions.
# Nitsche's method is a technique for imposing boundary conditions weakly, which can be particularly useful for problems with complex boundary conditions.
# The parameters provided are used to calculate the stabilization terms:
# - $s_{\kappa} = 1$ is a parameter related to the boundary condition.
# - $a = 200$ is the advection coefficient.
# - $V_{\kappa} = Ch^{-1}(1+|s_{\kappa}|)$ is a stabilization parameter, where $C = 50$.
# - $\tau = \frac{h}{2|a|}(coth(P^e)-\frac{1}{P^e})$ is the SUPG stabilization parameter, where $P^e = \frac{|a|h}{2\kappa}$ is the element Peclet number.
# These terms are used to adjust the mass matrix A and the right-hand side vector b to improve the stability and accuracy of the numerical solution.

def stabilization(A, b):
    '''Inputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    Outputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector 1d array of any size, 1d array of shape (M,)
    '''

    
    M = A.shape[0]
    h = 1.0 / (M - 1)  # Assuming a uniform grid over the domain [0, 1]
    a = 200
    C = 50
    s_kappa = 1
    kappa = 1  # Assuming a constant diffusion coefficient for simplicity
    
    # Calculate V_kappa
    V_kappa = C * h**-1 * (1 + abs(s_kappa))
    
    # Loop over each element to adjust A and b
    for i in range(M - 1):
        # Calculate the element Peclet number
        P_e = abs(a) * h / (2 * kappa)
        
        # Calculate the SUPG stabilization parameter tau
        if P_e != 0:
            tau = (h / (2 * abs(a))) * (1 / np.tanh(P_e) - 1 / P_e)
        else:
            tau = 0
        
        # Adjust the mass matrix A and right-hand side vector b
        # Nitsche term and SUPG stabilization term
        A[i, i] += V_kappa * h
        A[i, i+1] -= V_kappa * h
        A[i+1, i] -= V_kappa * h
        A[i+1, i+1] += V_kappa * h
        
        # SUPG stabilization
        A[i, i] += tau * h
        A[i, i+1] -= tau * h
        A[i+1, i] -= tau * h
        A[i+1, i+1] += tau * h
        
        # Adjust the right-hand side vector b
        b[i] += V_kappa * h
        b[i+1] -= V_kappa * h
    
    return A, b


from scicode.parse.parse import process_hdf5_to_tuple

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
