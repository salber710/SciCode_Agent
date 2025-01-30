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



# Background: 
# The task is to assemble a mass matrix A and a right-hand side vector b for a finite element method (FEM) problem.
# The mass matrix is a key component in FEM, representing the distribution of mass in the system.
# The right-hand side vector typically represents external forces or source terms.
# We will use a third-order Gaussian quadrature for numerical integration, which is a method to approximate the integral of a function.
# Gaussian quadrature is chosen for its accuracy in integrating polynomials of degree up to 2n-1, where n is the number of points.
# SUPG (Streamline Upwind Petrov-Galerkin) is a stabilization technique used to handle numerical instabilities in convection-dominated problems.
# The SUPG term modifies the test function to add stability to the numerical solution.


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

    # Define the third-order Gaussian quadrature points and weights
    gauss_points = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gauss_weights = np.array([5/9, 8/9, 5/9])

    # Define the element size
    h = 1.0 / (M - 1)

    # Loop over each element
    for i in range(M - 1):
        # Local element matrix and vector
        A_local = np.zeros((2, 2))
        b_local = np.zeros(2)

        # Loop over each quadrature point
        for gp, gw in zip(gauss_points, gauss_weights):
            # Map the quadrature point to the element
            xi = 0.5 * (gp + 1) * h + i * h

            # Evaluate shape functions and their derivatives at the quadrature point
            N1 = (xi - i * h) / h
            N2 = ((i + 1) * h - xi) / h
            dN1_dx = 1 / h
            dN2_dx = -1 / h

            # Compute the local contributions to the mass matrix
            A_local[0, 0] += gw * N1 * N1 * h
            A_local[0, 1] += gw * N1 * N2 * h
            A_local[1, 0] += gw * N2 * N1 * h
            A_local[1, 1] += gw * N2 * N2 * h

            # Compute the local contributions to the right-hand side vector
            # Assuming a source term f(x) = 1 for simplicity
            f_x = 1
            b_local[0] += gw * N1 * f_x * h
            b_local[1] += gw * N2 * f_x * h

            # SUPG stabilization term (simplified for demonstration)
            tau = h / 2  # Stabilization parameter
            A_local[0, 0] += tau * dN1_dx * dN1_dx * h
            A_local[0, 1] += tau * dN1_dx * dN2_dx * h
            A_local[1, 0] += tau * dN2_dx * dN1_dx * h
            A_local[1, 1] += tau * dN2_dx * dN2_dx * h

        # Assemble the local contributions into the global matrix and vector
        A[i:i+2, i:i+2] += A_local
        b[i:i+2] += b_local

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