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

    # Helper function to check if a distribution is valid
    def is_valid_distribution(dist):
        return abs(sum(dist) - 1.0) < 1e-12 and all(x >= 0 for x in dist)

    # Validate the input distributions
    if not is_valid_distribution(p) or not is_valid_distribution(q):
        raise ValueError("Input distributions must sum to 1 and have non-negative values.")

    # Initialize divergence
    divergence = 0.0

    # Iterate over the distributions using zip and enumerate for index
    for idx, (p_i, q_i) in enumerate(zip(p, q)):
        if p_i > 0:
            # Calculate log base 2 using the change of base formula with natural log
            log_base_2 = (p_i / q_i) if q_i != 0 else float('inf')
            divergence += p_i * (log_base_2).bit_length()  # Alternative approach to simulate log2

    return divergence / 2  # Adjust the bit_length calculation by dividing by 2


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