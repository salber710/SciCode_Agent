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



# Background: Mutual information is a measure from information theory that quantifies the amount of information obtained about one random variable through another random variable. 
# For a classical channel, the mutual information between an input random variable X and an output random variable Y is defined as:
# I(X; Y) = H(Y) - H(Y | X)
# Where H(Y) is the entropy of the output variable Y, and H(Y | X) is the conditional entropy of Y given X.
# The entropy H(Y) is calculated as H(Y) = - Σ p(y) * log2(p(y)) over all possible values of y.
# The conditional entropy H(Y | X) is calculated as H(Y | X) = - Σ p(y, x) * log2(p(y | x)) over all possible pairs (x, y).
# The joint probability p(y, x) is obtained from the channel matrix (p(y | x)) and the prior distribution (p(x)).
# The mutual information quantifies the reduction in uncertainty about the input given the output.

def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    
    # Import numpy dependency

    
    # Convert inputs to numpy arrays
    channel = np.asarray(channel)
    prior = np.asarray(prior)
    
    # Calculate the probability of each output y, p(y)
    # p(y) = Σ p(y | x) * p(x) for all x
    p_y = np.dot(prior, channel)
    
    # Calculate the joint probability p(y, x) = p(y | x) * p(x)
    # This can be done by element-wise multiplication of channel and prior, followed by reshaping
    p_y_given_x = channel.T * prior
    p_y_x = p_y_given_x.T
    
    # Calculate the mutual information I(X; Y)
    # I(X; Y) = Σ p(y, x) * log2(p(y | x) / p(y))
    #          = Σ p(y, x) * (log2(p(y | x)) - log2(p(y)))
    
    # Use masking to avoid log2(0)
    mask = p_y_x > 0
    mutual_info = np.sum(p_y_x[mask] * (np.log2(p_y_given_x[mask]) - np.log2(p_y[mask[0]])))
    
    return mutual_info

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
