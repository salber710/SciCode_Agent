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


    # Helper function to calculate single element contribution to KL divergence
    def kl_element(p_i, q_i):
        if p_i == 0:
            return 0
        return p_i * log2(p_i / q_i)

    # Validate the input distributions
    if abs(sum(p) - 1) > 1e-10 or abs(sum(q) - 1) > 1e-10:
        raise ValueError("Input distributions must sum to 1.")
    if any(x < 0 for x in p) or any(x < 0 for x in q):
        raise ValueError("Input distributions must have non-negative values.")
    
    # Calculate the KL divergence
    divergence = sum(kl_element(p_i, q_i) for p_i, q_i in zip(p, q))
    
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