import numpy as np

# Background: Kullback-Leibler divergence (KL-divergence) is a measure from information theory that quantifies how one probability distribution diverges from a second, expected probability distribution. For two discrete probability distributions p and q, the KL-divergence from q to p is defined as:
# KL(p || q) = Σ (p[i] * log2(p[i] / q[i])) for each element i where p[i] > 0. 
# It is important that both distributions, p and q, are defined over the same support and sum to 1. The KL-divergence is non-negative and is zero if and only if the distributions are identical.


def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''
    p = np.asarray(p)
    q = np.asarray(q)

    # Ensure probability distributions do not contain zeros where p is positive
    mask = p > 0
    # Calculate the KL-divergence using the definition
    divergence = np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    
    return divergence



# Background: Mutual information is a measure from information theory that quantifies the amount of information obtained about one random variable through another random variable. For a classical channel with input random variable X and output random variable Y, mutual information I(X; Y) measures the reduction in uncertainty of X given the knowledge of Y. It is defined as:
# I(X; Y) = Σ Σ P(x, y) log2(P(x, y) / (P(x)P(y)))
# where P(x, y) is the joint probability of X and Y, P(x) is the marginal probability of X, and P(y) is the marginal probability of Y. In the case of a classical channel, P(x, y) can be derived from the channel matrix and the prior distribution.


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    # Convert inputs to numpy arrays
    channel = np.asarray(channel)
    prior = np.asarray(prior)
    
    # Calculate the joint probability distribution P(x, y)
    joint_prob = channel * prior[:, np.newaxis]

    # Calculate the marginal probability P(y)
    marginal_y = joint_prob.sum(axis=0)

    # Calculate the mutual information I(X; Y)
    mutual = 0.0
    for x in range(len(prior)):
        for y in range(len(marginal_y)):
            if joint_prob[x, y] > 0:
                mutual += joint_prob[x, y] * np.log2(joint_prob[x, y] / (prior[x] * marginal_y[y]))
    
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
