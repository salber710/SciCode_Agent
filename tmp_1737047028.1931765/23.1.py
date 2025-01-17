import numpy as np



# Background: The Kullback-Leibler (KL) divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. It is a non-symmetric measure of the difference between two probability distributions p and q over the same variable. The KL divergence is defined as:
# KL(p || q) = Σ p(x) * log2(p(x) / q(x))
# where the sum is over all possible events x. It is important that both p and q are valid probability distributions, meaning they sum to 1 and all elements are non-negative. The log base 2 is used here, which is common in information theory, as it measures the divergence in bits.


def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    # Convert inputs to numpy arrays if they are not already
    p = np.array(p)
    q = np.array(q)
    
    # Calculate the KL divergence using the formula
    divergence = np.sum(p * np.log2(p / q))
    
    return divergence


from scicode.parse.parse import process_hdf5_to_tuple

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
