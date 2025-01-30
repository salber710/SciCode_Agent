import numpy as np

# Background: In the context of finite element analysis, shape functions are used to interpolate the solution
# over an element. For a 1D linear element, the shape functions are linear polynomials that define how the 
# solution varies within the element. The shape functions for a linear element are typically defined over 
# a reference element and then mapped to the actual element. In this problem, we have two types of shape 
# functions, ω^1(x) and ω^2(x), which are defined over specific intervals. ω^1(x) is defined for the interval 
# [x_{i-1}, x_i] and linearly interpolates from 0 to 1, while ω^2(x) is defined for the interval [x_i, x_{i+1}] 
# and linearly interpolates from 1 to 0. The parameter h represents the element size, and etype determines 
# which shape function to use.

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

    
    if h == 0:
        raise ZeroDivisionError("Element size h cannot be zero.")
    
    v = np.zeros_like(p, dtype=float)
    
    if etype == 1:
        # Calculate ω^1(x) = (x - x_{i-1}) / h for x in [x_{i-1}, x_i]
        x_i_minus_1 = i * h - h
        x_i = i * h
        for j in range(len(p)):
            if x_i_minus_1 <= p[j] <= x_i:
                v[j] = (p[j] - x_i_minus_1) / h
            else:
                v[j] = 0.0
    elif etype == 2:
        # Calculate ω^2(x) = (x_{i+1} - x) / h for x in [x_i, x_{i+1}]
        x_i = i * h
        x_i_plus_1 = i * h + h
        for j in range(len(p)):
            if x_i <= p[j] <= x_i_plus_1:
                v[j] = (x_i_plus_1 - p[j]) / h
            else:
                v[j] = 0.0
    else:
        # Handle invalid etype by returning zero array
        v.fill(0.0)
    
    return v



# Background: In finite element analysis, the mass matrix and the right-hand side vector are crucial components
# of the system of equations that need to be solved. The mass matrix represents the distribution of mass in the
# system, while the right-hand side vector represents external forces or sources. The assembly of these components
# involves integrating the shape functions over the domain. The third-order Gauss quadrature is a numerical 
# integration method that provides accurate results by evaluating the integrand at specific points (Gauss points)
# and weighting them appropriately. The SUPG (Streamline Upwind Petrov-Galerkin) term is a stabilization technique
# used to improve the numerical solution of advection-dominated problems by adding a small amount of artificial 
# diffusion in the direction of the flow.


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
    
    # Define the Gauss quadrature points and weights for third-order quadrature
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])
    
    # Element size (assuming uniform grid)
    h = 1.0 / (M - 1)
    
    # Loop over each element
    for i in range(M - 1):
        # Local mass matrix and right-hand side vector for the current element
        local_A = np.zeros((2, 2))
        local_b = np.zeros(2)
        
        # Loop over Gauss points
        for gp, gw in zip(gauss_points, gauss_weights):
            # Map Gauss point to the element
            xi = 0.5 * (1 + gp)  # xi in [0, 1]
            x = i * h + xi * h  # x in [x_i, x_{i+1}]
            
            # Evaluate shape functions and their derivatives at the Gauss point
            N1 = 1 - xi
            N2 = xi
            dN1_dx = -1 / h
            dN2_dx = 1 / h
            
            # SUPG stabilization term (simplified for demonstration)
            tau = h / 2  # Stabilization parameter
            
            # Contribution to the local mass matrix
            local_A[0, 0] += gw * N1 * N1 * h
            local_A[0, 1] += gw * N1 * N2 * h
            local_A[1, 0] += gw * N2 * N1 * h
            local_A[1, 1] += gw * N2 * N2 * h
            
            # Contribution to the local right-hand side vector
            # Assuming a source term f(x) = 1 for simplicity
            f = 1
            local_b[0] += gw * N1 * f * h
            local_b[1] += gw * N2 * f * h
            
            # SUPG stabilization contribution
            local_b[0] += gw * tau * dN1_dx * f * h
            local_b[1] += gw * tau * dN2_dx * f * h
        
        # Assemble local contributions into the global matrix and vector
        A[i:i+2, i:i+2] += local_A
        b[i:i+2] += local_b
    
    return A, b

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('54.2', 3)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
M = 11
assert cmp_tuple_or_list(assemble(M), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
M = 23
assert cmp_tuple_or_list(assemble(M), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
M = 35
assert cmp_tuple_or_list(assemble(M), target)
