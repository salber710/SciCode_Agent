from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    # Import the log function from the math module


    # Function to validate if the distributions are proper
    def validate_distribution(dist):
        if abs(sum(dist) - 1.0) > 1e-9:
            return False
        if any(x < 0 for x in dist):
            return False
        return True

    # Validate both p and q
    if not (validate_distribution(p) and validate_distribution(q)):
        raise ValueError("Input distributions must sum to 1 and have non-negative values.")

    # Calculate the KL-divergence using a generator expression and sum
    divergence = sum(
        (p_i * log(p_i / q_i, 2) if p_i > 0 else 0) for p_i, q_i in zip(p, q)
    )

    return divergence


try:
    targets = process_hdf5_to_tuple('23.1', 3)
    target = targets[0]
    p = [1/2,1/2,0,0]
    q = [1/4,1/4,1/4,1/4]
    assert np.allclose(KL_divergence(p,q), target)

    target = targets[1]
    p = [1/2,1/2]
    q = [1/2,1/2]
    assert np.allclose(KL_divergence(p,q), target)

    target = targets[2]
    p = [1,0]
    q = [1/4,3/4]
    assert np.allclose(KL_divergence(p,q), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e