from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


def KL_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Validate inputs
    assert np.all(p >= 0) and np.all(q >= 0), "Probabilities must be non-negative"
    assert np.isclose(np.sum(p), 1), "p must sum to 1"
    assert np.isclose(np.sum(q), 1), "q must sum to 1"

    # Compute KL divergence using a masked array to handle zero probabilities
    mask = (p > 0) & (q > 0)
    p_masked = np.ma.masked_array(p, mask=~mask)
    q_masked = np.ma.masked_array(q, mask=~mask)
    divergence = np.ma.sum(p_masked * np.ma.log2(p_masked / q_masked))

    return divergence



# Background: Mutual information is a measure of the amount of information that one random variable contains about another random variable. 
# In the context of a communication channel, it quantifies how much information the output of the channel reveals about the input. 
# The mutual information I(X;Y) between an input random variable X and an output random variable Y is defined as:
# I(X;Y) = H(Y) - H(Y|X), where H(Y) is the entropy of Y and H(Y|X) is the conditional entropy of Y given X.
# Alternatively, it can be expressed as I(X;Y) = sum_x sum_y P(x, y) log2(P(x, y) / (P(x)P(y))),
# where P(x, y) is the joint probability of x and y, P(x) is the probability of x, and P(y) is the probability of y.
# In a channel context, P(y|x) is given by the channel matrix, and P(x) is the prior distribution.


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    # Calculate the joint probability distribution P(x, y)
    joint_prob = channel * prior[:, np.newaxis]

    # Calculate the marginal probability distribution P(y)
    marginal_y = np.sum(joint_prob, axis=0)

    # Calculate the mutual information
    mutual = 0.0
    for i in range(len(prior)):
        for j in range(len(marginal_y)):
            if joint_prob[i, j] > 0:  # Only consider non-zero probabilities
                mutual += joint_prob[i, j] * np.log2(joint_prob[i, j] / (prior[i] * marginal_y[j]))

    return mutual


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e