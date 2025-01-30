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
    
    num_inputs = len(prior)
    num_outputs = len(channel[0])

    # Calculate joint distribution P(i, j) = P(j|i) * P(i)
    joint_distribution = np.zeros((num_inputs, num_outputs))
    for i in range(num_inputs):
        for j in range(num_outputs):
            joint_distribution[i, j] = prior[i] * channel[i][j]

    # Calculate the marginal distribution of the output P(j) = sum_i P(i, j)
    marginal_output_distribution = np.sum(joint_distribution, axis=0)

    # Calculate mutual information using I(X; Y) = sum_i sum_j P(i, j) * log(P(i, j) / (P(i) * P(j)))
    mutual_info_value = 0.0
    for i in range(num_inputs):
        for j in range(num_outputs):
            if joint_distribution[i, j] > 0:
                term = joint_distribution[i, j] * np.log(joint_distribution[i, j] / (prior[i] * marginal_output_distribution[j]))
                mutual_info_value += term

    # Convert natural log to log base 2
    mutual_info_value /= np.log(2)

    return mutual_info_value




def blahut_arimoto(channel, e):
    num_inputs, num_outputs = channel.shape

    # Initialize the input distribution with a geometric sequence
    input_dist = np.geomspace(1, num_inputs, num=num_inputs)
    input_dist /= np.sum(input_dist)

    # Initialize variables for mutual information
    previous_capacity = -np.inf
    current_capacity = 0.0

    # Helper function to compute conditional probabilities
    def compute_conditional_probabilities(channel, input_dist):
        p_output = np.dot(input_dist, channel)
        return np.where(p_output > 0, channel / p_output, 0)

    # Iterate until the change in capacity is less than the error threshold
    while np.abs(current_capacity - previous_capacity) > e:
        previous_capacity = current_capacity

        # Compute the conditional probabilities P(j|i) / P(j)
        conditional_probs = compute_conditional_probabilities(channel, input_dist)

        # Update the input distribution
        input_dist = np.array([np.exp(np.sum(channel[i] * np.log2(conditional_probs[i]))) for i in range(num_inputs)])
        input_dist /= np.sum(input_dist)

        # Compute the new mutual information
        current_capacity = 0.0
        for i in range(num_inputs):
            for j in range(num_outputs):
                if channel[i, j] > 0:
                    current_capacity += input_dist[i] * channel[i, j] * np.log2(conditional_probs[i, j])

    return current_capacity


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e