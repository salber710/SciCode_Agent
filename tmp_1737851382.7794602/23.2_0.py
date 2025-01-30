import numpy as np

# Background: The Kullback-Leibler (KL) divergence is a measure of how one probability distribution diverges from a second, expected probability distribution. It is a non-symmetric measure of the difference between two probability distributions p and q. The KL divergence is defined only if for all i, q[i] = 0 implies p[i] = 0 (absolute continuity). The formula for KL divergence is:
# KL(p || q) = Σ p[i] * log2(p[i] / q[i])
# where the sum is over all possible events. In practice, this means iterating over the elements of the distributions p and q, computing the term for each element, and summing them up. It is important to handle cases where p[i] is 0, as 0 * log(0/q[i]) is defined to be 0.


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
    
    # Check for equal length of distributions
    if len(p) != len(q):
        raise ValueError("p and q must be of the same length")
    
    # Check for absolute continuity
    if np.any((q == 0) & (p != 0)):
        raise ValueError("q[i] = 0 implies p[i] must be 0 for all i")
    
    # Calculate the KL divergence
    divergence = np.sum(np.where(p != 0, p * np.log2(p / q), 0))
    
    return divergence



# Background: Mutual information is a measure of the amount of information that one random variable contains about another random variable. It quantifies the reduction in uncertainty about one random variable given knowledge of the other. For a classical channel with input random variable X and output random variable Y, the mutual information I(X; Y) is defined as:
# I(X; Y) = Σ Σ P(x, y) * log2(P(x, y) / (P(x) * P(y)))
# where P(x, y) is the joint probability of X = x and Y = y, P(x) is the probability of X = x, and P(y) is the probability of Y = y. The joint probability P(x, y) can be calculated as P(x) * P(y|x), where P(y|x) is the conditional probability given by the channel matrix. The mutual information can also be expressed as the difference between the entropy of the output and the conditional entropy of the output given the input.


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    # Calculate the joint probability distribution P(x, y)
    joint_prob = np.outer(prior, channel)
    
    # Calculate the marginal probability distribution P(y)
    marginal_y = np.sum(joint_prob, axis=0)
    
    # Calculate the mutual information
    mutual = 0.0
    for i in range(len(prior)):
        for j in range(len(marginal_y)):
            if joint_prob[i, j] > 0:
                mutual += joint_prob[i, j] * np.log2(joint_prob[i, j] / (prior[i] * marginal_y[j]))
    
    return mutual

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('23.2', 3)
target = targets[0]

channel = np.eye(2)
prior = [0.5,0.5]
assert np.allclose(mutual_info(channel, prior), target)
target = targets[1]

channel = np.array([[1/2,1/2],[1/2,1/2]])
prior = [3/8,5/8]
assert np.allclose(mutual_info(channel, prior), target)
target = targets[2]

channel = np.array([[0.8,0],[0,0.8],[0.2,0.2]])
prior = [1/2,1/2]
assert np.allclose(mutual_info(channel, prior), target)
