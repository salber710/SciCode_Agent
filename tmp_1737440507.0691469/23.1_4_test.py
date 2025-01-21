import numpy as np



# Background: The Kullback-Leibler (KL) divergence is a measure from the field of information theory that quantifies the difference between two probability distributions. Specifically, it measures how one probability distribution (p) diverges from a second, reference probability distribution (q). The KL divergence is calculated as the sum over the support of p of p(i) * log2(p(i) / q(i)). It is important that the probability distributions p and q are defined on the same support, and q(i) should be non-zero wherever p(i) is non-zero. The base of the logarithm used here is 2, making the measure in bits.

def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''


    # Convert inputs to numpy arrays if they aren't already
    p = np.array(p)
    q = np.array(q)
    
    # Ensure q does not contain zero where p is non-zero to avoid division by zero
    # Adding a small epsilon to avoid log(0) in numerical computation
    epsilon = 1e-10
    q = np.where(q == 0, epsilon, q)
    
    # Calculate the KL divergence
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
