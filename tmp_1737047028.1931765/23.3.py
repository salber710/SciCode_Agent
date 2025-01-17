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


# Background: Mutual information is a measure of the amount of information that one random variable contains about another random variable. In the context of a communication channel, it quantifies the amount of information that the output random variable (the received message) contains about the input random variable (the sent message). The mutual information I(X;Y) between two random variables X and Y is defined as:
# I(X;Y) = Σ Σ p(x, y) * log2(p(x, y) / (p(x) * p(y)))
# where p(x, y) is the joint probability distribution of X and Y, p(x) is the marginal probability distribution of X, and p(y) is the marginal probability distribution of Y. In the context of a channel, the joint probability p(x, y) can be expressed as p(y|x) * p(x), where p(y|x) is the channel matrix and p(x) is the prior distribution of the input. The mutual information can also be expressed as the difference between the entropy of the output and the conditional entropy of the output given the input.


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    # Convert inputs to numpy arrays if they are not already
    channel = np.array(channel)
    prior = np.array(prior)
    
    # Calculate the joint probability distribution p(x, y) = p(y|x) * p(x)
    joint_prob = channel * prior[:, np.newaxis]
    
    # Calculate the marginal probability distribution of the output p(y)
    p_y = np.sum(joint_prob, axis=0)
    
    # Calculate the mutual information
    mutual = 0.0
    for i in range(len(prior)):
        for j in range(len(p_y)):
            if joint_prob[i, j] > 0:  # To avoid log(0) issues
                mutual += joint_prob[i, j] * np.log2(joint_prob[i, j] / (prior[i] * p_y[j]))
    
    return mutual



# Background: The Blahut-Arimoto algorithm is an iterative method used to compute the channel capacity of a discrete memoryless channel. The channel capacity is the maximum mutual information between the input and output of the channel, optimized over all possible input distributions. The algorithm starts with an initial guess for the input distribution and iteratively refines it to maximize the mutual information. The process continues until the change in the estimated rate (mutual information) between iterations is less than a specified error threshold. The channel capacity is reached when this convergence criterion is met.


def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''
    # Number of input and output symbols
    num_inputs, num_outputs = channel.shape
    
    # Initialize the input distribution uniformly
    p_x = np.full(num_inputs, 1.0 / num_inputs)
    
    # Initialize the rate
    rate_new = 0.0
    rate_old = -np.inf
    
    # Iterate until convergence
    while True:
        # Calculate the joint distribution p(x, y) = p(y|x) * p(x)
        joint_prob = channel * p_x[:, np.newaxis]
        
        # Calculate the marginal distribution of the output p(y)
        p_y = np.sum(joint_prob, axis=0)
        
        # Calculate the conditional distribution p(x|y)
        p_x_given_y = joint_prob / p_y
        
        # Update the input distribution p(x)
        p_x = np.exp(np.sum(p_x_given_y * np.log2(channel), axis=1))
        p_x /= np.sum(p_x)  # Normalize to ensure it sums to 1
        
        # Calculate the new rate (mutual information)
        rate_new = np.sum(p_x * np.sum(channel * np.log2(p_x_given_y), axis=1))
        
        # Check for convergence
        if np.abs(rate_new - rate_old) < e:
            break
        
        # Update the old rate
        rate_old = rate_new
    
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
