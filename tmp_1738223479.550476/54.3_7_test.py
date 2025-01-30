from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

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

    # Calculate the boundaries for the intervals
    x_prev = (i - 1) * h
    x_curr = i * h
    x_next = (i + 1) * h

    # Initialize an empty list for storing the results
    v = []

    # Define a nested function to handle each interval calculation
    def compute_value(x, etype):
        if etype == 1:
            return (x - x_prev) / h if x_prev <= x <= x_curr else 0.0
        elif etype == 2:
            return (x_next - x) / h if x_curr <= x <= x_next else 0.0
        else:
            return 0.0

    # Apply the nested function to each point in p
    for x in p:
        v.append(compute_value(x, etype))
    
    return v



def assemble(M):
    '''Inputs:
    M : number of grid, integer
    Outputs:
    A: mass matrix, 2d array size M*M
    b: right hand side vector , 1d array size M*1
    '''

    # Define the third-order Gauss quadrature points and weights
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])

    # Initialize the mass matrix A and right-hand side vector b
    A = np.zeros((M, M))
    b = np.zeros(M)

    # Use a quadratic function to distribute nodes
    nodes = (np.linspace(0, 1, M))**2
    h = np.diff(nodes)

    # Define a SUPG parameter that varies with the square of the element length
    def calc_tau(h_elem, scale=3.0):
        return h_elem**2 / (scale + h_elem)

    # Loop over each element to assemble the matrix and vector
    for elem in range(M - 1):
        # Length of the current element
        h_elem = h[elem]

        # Elemental mass matrix and vector
        Ae = np.zeros((2, 2))
        be = np.zeros(2)

        # Integrate over each Gauss point
        for gp, weight in zip(gauss_points, gauss_weights):
            # Map Gauss point to the element domain
            xi = (gp + 1) / 2
            x = (1 - xi) * nodes[elem] + xi * nodes[elem + 1]

            # Evaluate shape functions and their derivatives at the Gauss point
            N = np.array([(1 - xi), xi])
            dN_dx = np.array([-1 / h_elem, 1 / h_elem])

            # Compute elemental mass matrix contribution using a quadratic scale
            Ae += weight * np.outer(N, N) * h_elem / 8

            # SUPG stabilization parameter
            tau = calc_tau(h_elem)

            # Compute elemental right hand side vector with SUPG stabilization
            be += weight * (N * np.sqrt(x) + tau * np.dot(dN_dx, N)) * h_elem / 8

        # Assemble elemental contributions into global matrix and vector
        A[elem:elem+2, elem:elem+2] += Ae
        b[elem:elem+2] += be

    return A, b




def stabilization(A, b):
    '''Inputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    Outputs:
    A : mass matrix, 2d array of shape (M,M)
    b : right hand side vector, 1d array of shape (M,)
    '''

    M = A.shape[0]
    s_kappa = 1
    a = 200
    C = 50
    kappa = 1e-5  # Assume a small diffusion coefficient

    # Calculate element length with a different assumption
    h_elem = 1.0 / np.sqrt(M + 4)  # Use a square root function to vary the grid spacing

    # Calculate V_kappa using an alternative approach
    V_kappa = C * np.log1p(1 / h_elem) * (1 + abs(s_kappa))

    # Calculate Peclet number P^e
    Pe = abs(a) * h_elem / (2 * kappa)

    # Calculate tau with a modified formula
    tau = (h_elem / (2 * abs(a))) * (1 / np.tanh(Pe) - 1 / (Pe + 1e-10))  # Add a small constant to avoid singularity

    # Define a new pattern for distributing SUPG contributions
    SUPG_matrix = np.zeros((M, M))

    # Use a Fibonacci pattern for distribution
    fib = [0, 1]
    while len(fib) < M:
        fib.append(fib[-1] + fib[-2])

    for i in range(M):
        for j in range(M):
            index_sum = fib[(i + j) % len(fib)]
            SUPG_matrix[i, j] = tau * V_kappa * (index_sum % 5) / 5.0

    # Update A with the SUPG_matrix contributions
    A += SUPG_matrix

    # Adjust the right-hand side vector using an exponential pattern
    exp_pattern = np.exp(-np.linspace(0, 1, M))
    b += np.dot(SUPG_matrix, exp_pattern)

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