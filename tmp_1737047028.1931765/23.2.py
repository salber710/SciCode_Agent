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
