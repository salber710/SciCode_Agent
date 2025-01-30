from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def Bspline(xi, i, p, Xi):
    '''Inputs:
    xi : knot index, integer
    i : polynomial index , integer
    p : polynomial degree of basis function , integer
    Xi : knot vector, 1d array of arbitrary size
    Outputs:
    1d array of size 1, 2 or 3
    '''
    # Define a helper function that iteratively computes the basis function
    def iterative_basis(xi, i, p, Xi):
        # Initialize a table to store values of the basis functions
        N = [[0.0] * (p + 1) for _ in range(len(Xi) - 1)]
        
        # Base case: p = 0
        for j in range(len(Xi) - 1):
            N[j][0] = 1.0 if Xi[j] <= xi < Xi[j + 1] else 0.0

        # Fill the table iteratively for higher degrees
        for q in range(1, p + 1):
            for j in range(len(Xi) - q - 1):
                # Compute coefficients
                if Xi[j + q] != Xi[j]:
                    alpha = (xi - Xi[j]) / (Xi[j + q] - Xi[j])
                else:
                    alpha = 0.0

                if Xi[j + q + 1] != Xi[j + 1]:
                    beta = (Xi[j + q + 1] - xi) / (Xi[j + q + 1] - Xi[j + 1])
                else:
                    beta = 0.0

                # Compute the value of the basis function using previously computed values
                N[j][q] = alpha * N[j][q - 1] + beta * N[j + 1][q - 1]

        return N[i][p]

    # Evaluate the basis function using the iterative approach
    return iterative_basis(xi, i, p, Xi)


try:
    targets = process_hdf5_to_tuple('18.1', 3)
    target = targets[0]
    xi =0.1
    i = 1
    p = 2
    Xi = [0,0,0,1,1,1]
    assert np.allclose(Bspline(xi, i, p, Xi), target)

    target = targets[1]
    xi = 1.5
    i = 1
    p = 3
    Xi = [0,0,0,1,1,1,2,2,2]
    assert np.allclose(Bspline(xi, i, p, Xi), target)

    target = targets[2]
    xi = 0.5
    i = 1
    p = 3
    Xi = [0,0,1,1,2,2,3,3]
    assert np.allclose(Bspline(xi, i, p, Xi), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e