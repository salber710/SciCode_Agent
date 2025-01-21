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



# Background: The Blahut-Arimoto algorithm is an iterative method used to compute the channel capacity of a discrete memoryless channel. The channel capacity is the maximum mutual information between the input and output of the channel, optimized over all possible input distributions. The algorithm starts with an initial input distribution and iteratively updates it to maximize the mutual information. At each step, the input distribution is updated based on the conditional distribution of the output given the input, calculated from the channel matrix. The process continues until the change in mutual information (or rate) between iterations falls below a specified error threshold.


def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''
    channel = np.asarray(channel)
    num_inputs, num_outputs = channel.shape

    # Initialize the input distribution (uniformly)
    prior = np.ones(num_inputs) / num_inputs

    # Initialize the rate
    rate_old = 0.0

    while True:
        # Compute Q(y) - the marginal probability of the outputs
        q_y = np.zeros(num_outputs)
        for y in range(num_outputs):
            q_y[y] = np.sum(prior[x] * channel[x, y] for x in range(num_inputs))

        # Update the input distribution
        new_prior = np.zeros(num_inputs)
        for x in range(num_inputs):
            new_prior[x] = 0
            for y in range(num_outputs):
                if channel[x, y] > 0:
                    new_prior[x] += channel[x, y] * np.log2(channel[x, y] / q_y[y])
            new_prior[x] = np.exp2(new_prior[x])

        # Normalize the new input distribution
        new_prior /= np.sum(new_prior)

        # Calculate the new rate
        rate_new = 0.0
        for x in range(num_inputs):
            for y in range(num_outputs):
                if channel[x, y] > 0:
                    rate_new += new_prior[x] * channel[x, y] * np.log2(channel[x, y] / q_y[y])

        # Check for convergence
        if abs(rate_new - rate_old) < e:
            break

        # Update the old rate and prior for the next iteration
        rate_old = rate_new
        prior = new_prior

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
