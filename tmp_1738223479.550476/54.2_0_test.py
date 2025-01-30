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



# Background: 
# In finite element analysis, assembling the mass matrix and the right-hand side vector is essential for solving partial differential equations numerically. 
# The mass matrix A represents the discretized version of the integral of the product of shape functions over each element. 
# The right-hand side vector b typically represents source terms or boundary conditions.
# To accurately compute these integrals, we use numerical integration techniques such as Gauss quadrature. 
# For the third-order Gauss quadrature, we use specific weights and points to approximate the integral over an element.
# The SUPG (Streamline Upwind/Petrov-Galerkin) stabilization technique is used to improve the accuracy of solutions, especially in advection-dominated problems, by modifying the weight functions.


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

    # Assume a uniform grid spacing
    h = 1.0 / (M - 1)

    # Loop over each element to assemble the matrix and vector
    for i in range(M - 1):
        # Elemental mass matrix and vector
        Ae = np.zeros((2, 2))
        be = np.zeros(2)

        # Loop over each Gauss point
        for gp, weight in zip(gauss_points, gauss_weights):
            # Map Gauss point to the element domain
            x = (0.5 * (gp + 1) + i) * h
            
            # Evaluate shape functions (N1, N2) and their derivatives at the Gauss point
            N1 = (1 - gp) / 2
            N2 = (1 + gp) / 2
            dN1_dx = -1 / h
            dN2_dx = 1 / h

            # Compute elemental mass matrix contribution
            Ae[0, 0] += weight * N1 * N1 * h / 2
            Ae[0, 1] += weight * N1 * N2 * h / 2
            Ae[1, 0] += weight * N2 * N1 * h / 2
            Ae[1, 1] += weight * N2 * N2 * h / 2
            
            # SUPG stabilization (simplified)
            tau = h / 2
            be[0] += weight * N1 * x * h / 2 + tau * dN1_dx * N1 * h / 2
            be[1] += weight * N2 * x * h / 2 + tau * dN2_dx * N2 * h / 2
        
        # Assemble elemental contributions into global matrix and vector
        A[i:i+2, i:i+2] += Ae
        b[i:i+2] += be
    
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