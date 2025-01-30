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
    # Use a queue for iterative computation with breadth-first traversal


    # Initialize queue with the initial state
    queue = deque([(i, p)])
    results = {}

    while queue:
        current_i, current_p = queue.popleft()

        if (current_i, current_p) in results:
            continue

        if current_p == 0:
            # Base case: directly compute and store the result
            results[(current_i, current_p)] = 1.0 if Xi[current_i] <= xi < Xi[current_i + 1] else 0.0
        else:
            # Determine dependencies
            left_key = (current_i, current_p - 1)
            right_key = (current_i + 1, current_p - 1)

            # Add dependencies to the queue if not already processed
            if left_key not in results:
                queue.append(left_key)
            if right_key not in results:
                queue.append(right_key)

            # Only process current node if dependencies are resolved
            if left_key in results and right_key in results:
                # Calculate the coefficients
                denom1 = Xi[current_i + current_p] - Xi[current_i]
                denom2 = Xi[current_i + current_p + 1] - Xi[current_i + 1]

                alpha = (xi - Xi[current_i]) / denom1 if denom1 != 0 else 0.0
                beta = (Xi[current_i + current_p + 1] - xi) / denom2 if denom2 != 0 else 0.0

                # Use stored results for recursive calculation
                left_result = results[left_key]
                right_result = results[right_key]

                # Store the result for the current pair
                results[(current_i, current_p)] = alpha * left_result + beta * right_result

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