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
    
    if not isinstance(M, int):
        raise TypeError("Input M must be an integer.")
    if M <= 0:
        raise ValueError("Input M must be a positive integer greater than zero.")
    
    # Initialize the mass matrix A and the right-hand side vector b
    A = np.zeros((M, M))
    b = np.zeros(M)
    
    if M == 1:
        return A, b  # For a single grid point, the matrix and vector remain zero as no elements exist to integrate over.
    
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
    
    # Ensure the matrix A and vector b are symmetric and consistent at boundaries
    A[-1, -1] += A[0, 0]
    b[-1] += b[0]
    A[0, 0] = A[-1, -1]
    b[0] = b[-1]
    
    return A, b


# Background: In finite element methods, stabilization techniques such as the SUPG (Streamline Upwind Petrov-Galerkin)
# and Nitsche's method are used to improve the numerical solution of problems, especially those involving advection-dominated
# processes. The SUPG method adds a stabilization term to the weak form of the equations, which helps to prevent numerical
# oscillations by introducing artificial diffusion in the direction of the flow. Nitsche's method is used to impose boundary
# conditions weakly, which can be particularly useful for problems with complex boundary conditions. The parameters involved
# in these methods, such as the element Peclet number (P^e) and the stabilization parameter (tau), are crucial for determining
# the amount of stabilization applied. The element Peclet number is a dimensionless number that characterizes the relative
# importance of advection and diffusion. The stabilization parameter tau is derived from the Peclet number and helps to
# control the amount of artificial diffusion added.

def stabilization(A, b):
    '''Inputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    Outputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector 1d array of any size, 1d array of shape (M,)
    '''


    # Constants
    s_kappa = 1
    a = 200
    C = 50
    kappa = 1  # Assuming kappa is given or can be set to a default value

    M = A.shape[0]
    if M == 1:
        h = 1  # Handle single element case
    else:
        h = 1.0 / (M - 1)  # Assuming uniform grid

    # Check if dimensions of A and b are compatible
    if A.shape[1] != M or len(b) != M:
        raise ValueError("Dimensions of A and b are not compatible.")

    # Loop over each element
    for i in range(M - 1):
        # Calculate element Peclet number
        P_e = abs(a) * h / (2 * kappa)
        
        # Calculate stabilization parameter tau
        if P_e != 0:
            tau = (h / (2 * abs(a))) * (1 / np.tanh(P_e) - 1 / P_e)
        else:
            tau = 0
        
        # Calculate V_kappa
        V_kappa = C * h**-1 * (1 + abs(s_kappa))
        
        # Adjust mass matrix A and right-hand side vector b
        # Nitsche term and SUPG stabilization term
        A[i, i] += V_kappa * tau
        A[i, i+1] -= V_kappa * tau
        A[i+1, i] -= V_kappa * tau
        A[i+1, i+1] += V_kappa * tau
        
        b[i] += V_kappa * tau
        b[i+1] -= V_kappa * tau

    # Ensure the last element of b is adjusted correctly
    if M > 1:
        b[-1] += V_kappa * tau

    return A, b



# Background: In numerical linear algebra, solving a linear system of equations is a fundamental task. 
# The system is typically represented in matrix form as Ax = b, where A is a matrix, x is the vector of 
# unknowns, and b is the right-hand side vector. In the context of finite element analysis, A is often 
# the mass matrix, and b is the vector representing external forces or sources. Solving this system 
# involves finding the vector x that satisfies the equation. This can be done using various numerical 
# methods, with direct methods like Gaussian elimination or LU decomposition being common for small to 
# medium-sized systems. In Python, the NumPy library provides efficient routines for solving such systems 
# using functions like `np.linalg.solve`, which is based on LU decomposition and is suitable for dense 
# matrices.


def solve(N):
    '''Inputs: 
    N: number of element
    Outputs:
    sol: solution array, 1d array of size (N+1,)
    '''
    
    # Initialize the mass matrix A and the right-hand side vector b
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    
    # Assemble the mass matrix A and right-hand side vector b
    A, b = assemble(N+1)
    
    # Adjust the mass matrix A and right-hand side vector b with stabilization terms
    A, b = stabilization(A, b)
    
    # Solve the linear system Ax = b
    sol = np.linalg.solve(A, b)
    
    return sol

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('54.4', 6)
target = targets[0]

N = 32
assert np.allclose(solve(N), target)
target = targets[1]

N = 64
assert np.allclose(solve(N), target)
target = targets[2]

N = 8
assert np.allclose(solve(N), target)
target = targets[3]

def fexact(x, a, k):
    return 24/a*(k/a)**3 + 24/a*(k/a)**2*x + 12/a*(k/a)*x**2 + 4/a*x**3 + \
           (1 - 24/a*(k/a)**3 - 24/a*(k/a)**2 - 12/a*(k/a) - 4/a)/np.exp(a/k) * np.exp(a/k*x)
def L2Error(N, sol):    
    M = N+1
    h = 1/N
    err = 0.0
    gp = np.array([[-np.sqrt(3/5)], [0.0], [np.sqrt(3/5)]])
    gw = np.array([5/9, 8/9, 5/9])
    for e in range(1, N):
        for g in range(3):
            p = (2*e-1)*h/2 + gp[g]*h/2
            err = err + gw[g]*(sol[e]*basis(e,p,N+1,h,1) + sol[e+1]*basis(e+1,p,N+1,h,1) - fexact(p,200,1))**2
    err = err * h/2
    return err
N = 16
sol = solve(N)
assert np.allclose(L2Error(N,sol), target)
target = targets[4]

def fexact(x, a, k):
    return 24/a*(k/a)**3 + 24/a*(k/a)**2*x + 12/a*(k/a)*x**2 + 4/a*x**3 + \
           (1 - 24/a*(k/a)**3 - 24/a*(k/a)**2 - 12/a*(k/a) - 4/a)/np.exp(a/k) * np.exp(a/k*x)
def L2Error(N, sol):    
    M = N+1
    h = 1/N
    err = 0.0
    gp = np.array([[-np.sqrt(3/5)], [0.0], [np.sqrt(3/5)]])
    gw = np.array([5/9, 8/9, 5/9])
    for e in range(1, N):
        for g in range(3):
            p = (2*e-1)*h/2 + gp[g]*h/2
            err = err + gw[g]*(sol[e]*basis(e,p,N+1,h,1) + sol[e+1]*basis(e+1,p,N+1,h,1) - fexact(p,200,1))**2
    err = err * h/2
    return err
N = 32
sol = solve(N)
assert np.allclose(L2Error(N,sol), target)
target = targets[5]

def fexact(x, a, k):
    return 24/a*(k/a)**3 + 24/a*(k/a)**2*x + 12/a*(k/a)*x**2 + 4/a*x**3 + \
           (1 - 24/a*(k/a)**3 - 24/a*(k/a)**2 - 12/a*(k/a) - 4/a)/np.exp(a/k) * np.exp(a/k*x)
def L2Error(N, sol):    
    M = N+1
    h = 1/N
    err = 0.0
    gp = np.array([[-np.sqrt(3/5)], [0.0], [np.sqrt(3/5)]])
    gw = np.array([5/9, 8/9, 5/9])
    for e in range(1, N):
        for g in range(3):
            p = (2*e-1)*h/2 + gp[g]*h/2
            err = err + gw[g]*(sol[e]*basis(e,p,N+1,h,1) + sol[e+1]*basis(e+1,p,N+1,h,1) - fexact(p,200,1))**2
    err = err * h/2
    return err
N = 64
sol = solve(N)
assert np.allclose(L2Error(N,sol), target)
