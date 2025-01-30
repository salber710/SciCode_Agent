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
    # Define a lambda function to compute the coefficients
    alpha = lambda xi, i, p, Xi: (xi - Xi[i]) / (Xi[i + p] - Xi[i]) if Xi[i + p] != Xi[i] else 0.0
    beta = lambda xi, i, p, Xi: (Xi[i + p + 1] - xi) / (Xi[i + p + 1] - Xi[i + 1]) if Xi[i + p + 1] != Xi[i + 1] else 0.0

    # Iteratively compute basis functions using a stack without recursion
    stack = [(i, p)]
    results = {}

    while stack:
        current_i, current_p = stack.pop()

        if (current_i, current_p) in results:
            continue

        if current_p == 0:
            results[(current_i, current_p)] = 1.0 if Xi[current_i] <= xi < Xi[current_i + 1] else 0.0
        else:
            left_key = (current_i, current_p - 1)
            right_key = (current_i + 1, current_p - 1)

            if left_key not in results:
                stack.append((current_i, current_p))
                stack.append(left_key)
            elif right_key not in results:
                stack.append((current_i, current_p))
                stack.append(right_key)
            else:
                left_result = results[left_key]
                right_result = results[right_key]

                current_alpha = alpha(xi, current_i, current_p, Xi)
                current_beta = beta(xi, current_i, current_p, Xi)

                results[(current_i, current_p)] = current_alpha * left_result + current_beta * right_result

    return results[(i, p)]


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