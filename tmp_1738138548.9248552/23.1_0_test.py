from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: The Kullback-Leibler (KL) divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. It is a non-symmetric measure of the difference between two probability distributions p and q. The KL divergence is defined only if for all i, q[i] = 0 implies p[i] = 0 (absolute continuity). The formula for KL divergence is:
# KL(p || q) = Σ p[i] * log2(p[i] / q[i])
# where the sum is over all possible events. In practice, this means iterating over the elements of the distributions p and q, computing the term for each element, and summing them up. It is important to handle cases where p[i] is 0, as 0 * log(0) is defined to be 0.


def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    p = np.array(p)
    q = np.array(q)
    
    # Ensure that p and q are valid probability distributions
    assert np.all(p >= 0) and np.all(q >= 0), "Probabilities must be non-negative"
    assert np.isclose(np.sum(p), 1), "p must sum to 1"
    assert np.isclose(np.sum(q), 1), "q must sum to 1"
    
    # Calculate KL divergence
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