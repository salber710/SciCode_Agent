from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: KL-divergence (Kullback-Leibler divergence) is a measure of how one probability distribution diverges from a second, expected probability distribution. 
# It is often used in information theory to quantify the difference between two probability distributions. 
# The formula for KL-divergence from a probability distribution p to a distribution q is:
# KL(p || q) = Σ p(x) * log2(p(x) / q(x)) for all x where p(x) > 0.
# This function assumes that both p and q are probability distributions over the same support,
# meaning they both sum to 1 and have non-negative values.


def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    p = np.array(p)
    q = np.array(q)
    
    # Ensure that p and q are both valid probability distributions
    if not np.isclose(np.sum(p), 1) or not np.isclose(np.sum(q), 1):
        raise ValueError("Input distributions must sum to 1.")
    
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Input distributions must have non-negative values.")
    
    # Compute the KL-divergence using only the elements where p > 0
    divergence = np.sum(np.where(p != 0, p * np.log2(p / q), 0))
    
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