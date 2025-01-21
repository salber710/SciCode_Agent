import numpy as np



# Background: The Kullback-Leibler divergence (KL-divergence) is a measure of how one probability distribution diverges from a second, expected probability distribution. It is often used in statistics and machine learning to measure the difference between two probability distributions. The KL-divergence of a distribution Q from a distribution P is defined as:
# 
# KL(P || Q) = Σ [ P(x) * log2(P(x) / Q(x)) ]
# 
# where the sum is over all events x in the support of P and Q. It is important to note that KL-divergence is not symmetric, meaning that KL(P || Q) is not necessarily equal to KL(Q || P). It is also not a true distance metric since it does not satisfy the triangle inequality and is not symmetric. In calculating KL-divergence, it is assumed that both distributions, P and Q, are properly normalized (i.e., sum to 1), and that Q(x) > 0 wherever P(x) > 0 to avoid divisions by zero or taking logarithms of zero.


def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # Avoid division by zero or log of zero by using a small value where q is zero
    q = np.where(q == 0, np.finfo(float).eps, q)
    
    # Calculate the KL-divergence
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
