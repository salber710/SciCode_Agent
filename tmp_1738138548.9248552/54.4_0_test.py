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



def stabilization(A, b):
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
    tau = h / (2 * abs(a)) * (1 / np.tanh(P_e) - 1 / P_e)
    
    # Adjust the mass matrix A and right hand side vector b
    for i in range(M):
        # Nitsche term contribution
        if i < M - 1:
            A[i, i] += V_kappa
            A[i + 1, i + 1] += V_kappa
            A[i, i + 1] -= V_kappa
            A[i + 1, i] -= V_kappa
        
        # SUPG stabilization term contribution
        A[i, i] += tau * a**2 * h
        
        # Adjust the right hand side vector b
        b[i] += V_kappa * s_kappa * h
    
    return A, b



# Background: Solving a linear system of equations is a fundamental task in numerical analysis and scientific computing.
# The system of equations is typically represented in matrix form as Ax = b, where A is a matrix, x is the vector of
# unknowns, and b is the right-hand side vector. The solution involves finding the vector x that satisfies this equation.
# In the context of finite element methods, A is the assembled global matrix, and b is the global right-hand side vector.
# The solution can be obtained using numerical linear algebra techniques, such as Gaussian elimination or iterative methods.
# Here, we will use numpy's linear algebra solver to find the solution vector x.

def solve(N):
    '''Inputs: 
    N: number of element
    Outputs:
    sol: solution array, 1d array of size (N+1,)
    '''

    
    # Define the number of nodes
    M = N + 1
    
    # Assemble the mass matrix A and right-hand side vector b
    A, b = assemble(M)
    
    # Adjust the mass matrix A and right-hand side vector b with stabilization terms
    A, b = stabilization(A, b)
    
    # Solve the linear system Ax = b
    sol = np.linalg.solve(A, b)
    
    return sol


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e