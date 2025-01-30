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



def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of output j given input i
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''
    
    # Get number of inputs and outputs
    num_inputs = len(prior)
    num_outputs = len(channel[0])
    
    # Calculate joint probabilities P(i, j) = P(i) * P(j|i)
    joint_prob = [[prior[i] * channel[i][j] for j in range(num_outputs)] for i in range(num_inputs)]
    
    # Calculate the marginal probabilities of the output P(j) = sum_i P(i, j)
    output_prob = [sum(joint_prob[i][j] for i in range(num_inputs)) for j in range(num_outputs)]
    
    # Calculate the mutual information I(X; Y)
    mutual_information = sum(
        joint_prob[i][j] * (log2(joint_prob[i][j]) - log2(prior[i] * output_prob[j]))
        for i in range(num_inputs)
        for j in range(num_outputs)
        if joint_prob[i][j] > 0
    )
    
    return mutual_information


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