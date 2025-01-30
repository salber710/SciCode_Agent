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

    # Define a non-uniform grid using a sinusoidal function for node distribution
    nodes = 0.5 * (1 - np.cos(np.linspace(0, np.pi, M)))
    h = np.diff(nodes)

    # Function to calculate SUPG stabilization parameter based on local Peclet number
    def calc_tau(h_elem, peclet=1.0):
        return h_elem / (2 * peclet)

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

            # Compute elemental mass matrix contribution
            Ae += weight * np.outer(N, N) * h_elem / 2

            # SUPG stabilization parameter
            tau = calc_tau(h_elem)

            # Compute elemental right hand side vector with SUPG stabilization
            be += weight * (N * x + tau * np.dot(dN_dx, N)) * h_elem / 2

        # Assemble elemental contributions into global matrix and vector
        A[elem:elem+2, elem:elem+2] += Ae
        b[elem:elem+2] += be

    return A, b


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e