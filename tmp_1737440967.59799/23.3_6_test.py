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


# Background: Mutual information is a measure from information theory that quantifies the amount of information obtained about one random variable through another random variable. For a given channel, which represents the conditional probabilities of outputs given inputs, and a prior input distribution, the mutual information I(X; Y) between an input random variable X and the output random variable Y is defined as:
# I(X; Y) = Σ Σ (P(X = x, Y = y) * log2(P(X = x, Y = y) / (P(X = x) * P(Y = y))))
# where P(X = x, Y = y) is the joint probability distribution, P(X = x) is the prior probability, and P(Y = y) is the marginal probability of Y.
# The joint probability P(X = x, Y = y) is given by P(Y = y | X = x) * P(X = x), where P(Y = y | X = x) is derived from the channel matrix.
# The marginal probability P(Y = y) is computed by summing over all joint probabilities for a given Y = y.


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    channel = np.asarray(channel)
    prior = np.asarray(prior)
    
    # Calculate the joint probability distribution P(X = x, Y = y)
    joint_prob = channel * prior[:, np.newaxis]

    # Calculate the marginal probability distribution P(Y = y)
    marginal_y = np.sum(joint_prob, axis=0)

    # Calculate mutual information
    mutual = 0.0
    for x in range(len(prior)):
        for y in range(channel.shape[1]):
            if joint_prob[x, y] > 0:
                mutual += joint_prob[x, y] * np.log2(joint_prob[x, y] / (prior[x] * marginal_y[y]))

    return mutual



# Background: The Blahut-Arimoto algorithm is an iterative method used to compute the channel capacity of a discrete memoryless channel. Channel capacity is the maximum rate at which information can be reliably transmitted over a communication channel. The algorithm involves alternating optimization steps to maximize the mutual information between input and output distributions of the channel. In each iteration, the input distribution is updated to maximize the mutual information given the current output distribution, and vice versa. The process continues until the change in mutual information between iterations is below a predefined error threshold, indicating convergence.


def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''
    num_inputs, num_outputs = channel.shape
    
    # Initialize the input distribution uniformly
    p_x = np.full(num_inputs, 1.0 / num_inputs)
    
    # Initialize the rate
    rate_old = 0.0
    rate_new = float('inf')
    
    while abs(rate_new - rate_old) > e:
        rate_old = rate_new
        
        # Compute q(y), the intermediate output distribution
        q_y = np.dot(p_x, channel)
        
        # Update p(x|y), the new input distribution conditioned on output
        p_xy = channel / q_y
        p_xy = p_xy * p_x[:, np.newaxis]
        
        # Update p(x), the new input distribution
        p_x = np.exp(np.sum(p_xy * np.log2(channel / q_y), axis=1))
        p_x /= np.sum(p_x)
        
        # Recalculate q(y) with the updated p(x) for the rate
        q_y = np.dot(p_x, channel)
        
        # Calculate the new rate
        rate_new = np.sum(p_x @ (channel * np.log2(channel / q_y)))
    
    return rate_new

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('23.3', 7)
target = targets[0]

np.random.seed(0)
channel = np.array([[1,0,1/4],[0,1,1/4],[0,0,1/2]])
e = 1e-8
assert np.allclose(blahut_arimoto(channel,e), target)
target = targets[1]

np.random.seed(0)
channel = np.array([[0.1,0.6],[0.9,0.4]])
e = 1e-8
assert np.allclose(blahut_arimoto(channel,e), target)
target = targets[2]

np.random.seed(0)
channel = np.array([[0.8,0.5],[0.2,0.5]])
e = 1e-5
assert np.allclose(blahut_arimoto(channel,e), target)
target = targets[3]

np.random.seed(0)
bsc = np.array([[0.8,0.2],[0.2,0.8]])
assert np.allclose(blahut_arimoto(bsc,1e-8), target)
target = targets[4]

np.random.seed(0)
bec = np.array([[0.8,0],[0,0.8],[0.2,0.2]])
assert np.allclose(blahut_arimoto(bec,1e-8), target)
target = targets[5]

np.random.seed(0)
channel = np.array([[1,0,1/4],[0,1,1/4],[0,0,1/2]])
e = 1e-8
assert np.allclose(blahut_arimoto(channel,e), target)
target = targets[6]

np.random.seed(0)
bsc = np.array([[0.8,0.2],[0.2,0.8]])
assert np.allclose(blahut_arimoto(bsc,1e-8), target)
