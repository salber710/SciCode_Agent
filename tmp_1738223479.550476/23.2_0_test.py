from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def KL_divergence(p, q):
    '''Input
    p: probability distributions, 1-dimensional numpy array (or list) of floats
    q: probability distributions, 1-dimensional numpy array (or list) of floats
    Output
    divergence: KL-divergence of two probability distributions, a single scalar value (float)
    '''



    # Validate the input distributions
    def validate(dist):
        return abs(sum(dist) - 1.0) < 1e-12 and all(x >= 0 for x in dist)
    
    if not (validate(p) and validate(q)):
        raise ValueError("Input distributions must sum to 1 and have non-negative values.")
    
    # Use reduce to calculate the KL divergence
    divergence = reduce(
        lambda acc, pq: acc + (pq[0] * log2(pq[0] / pq[1]) if pq[0] > 0 else 0),
        zip(p, q),
        0.0
    )
    
    return divergence



# Background: Mutual information quantifies the amount of information obtained about one random variable through another random variable. 
# In the context of a communication channel, it measures the reduction in uncertainty about the input random variable given the output. 
# Mathematically, mutual information I(X; Y) between input X and output Y is given by:
# I(X; Y) = H(Y) - H(Y|X), where H(Y) is the entropy of the output, and H(Y|X) is the conditional entropy of the output given the input.
# Alternatively, it can be expressed as the expected KL-divergence between the posterior and prior distributions over the input given the output.


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of output j given input i
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''

    # Calculate the joint distribution P(i, j) = P(j|i) * P(i)
    joint = np.outer(prior, channel)

    # Calculate the marginal distribution P(j) = sum_i P(i, j)
    marginal_output = np.sum(joint, axis=0)

    # Calculate the mutual information I(X; Y)
    mutual = 0.0
    for i in range(len(prior)):
        for j in range(len(channel[0])):
            if joint[i, j] > 0:
                mutual += joint[i, j] * np.log2(joint[i, j] / (prior[i] * marginal_output[j]))

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