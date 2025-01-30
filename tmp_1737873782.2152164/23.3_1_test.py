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

    # Convert p and q to numpy arrays if they are not already
    p = np.array(p)
    q = np.array(q)

    # Ensure that the distributions are normalized
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute the KL-divergence
    divergence = np.sum(p * np.log2(p / q))

    return divergence


def mutual_info(channel, prior):
    '''Input
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of i given j
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''


    # Calculate the marginal distribution of the output variable
    output_marginal = np.dot(prior, channel)
    
    # Initialize the mutual information
    mutual = 0.0
    
    # Iterate over all input and output possibilities
    for i in range(len(prior)):  # Iterate over input states
        for j in range(len(output_marginal)):  # Iterate over output states
            if channel[i][j] > 0:  # Avoid log(0) which is undefined
                # Calculate the joint probability p(x, y)
                joint_prob = prior[i] * channel[i][j]
                
                # Calculate the mutual information increment
                mutual += joint_prob * np.log2(channel[i][j] / output_marginal[j])
    
    return mutual




def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''

    # Get the number of input and output states
    num_inputs, num_outputs = channel.shape
    
    # Initialize the input probability distribution uniformly
    input_prob = np.full(num_inputs, 1.0 / num_inputs)
    
    # Initialize rate variables
    rate_old = 0
    rate_new = float('inf')
    
    # Iteratively update the input probability distribution and calculate the rate
    while abs(rate_new - rate_old) > e:
        # Calculate the output distribution given the current input distribution
        output_prob = np.dot(input_prob, channel)
        
        # Update the rate_old
        rate_old = rate_new
        
        # Initialize a zero matrix for Q
        Q = np.zeros((num_inputs, num_outputs))
        
        # Update Q matrix values
        for i in range(num_inputs):
            for j in range(num_outputs):
                if output_prob[j] > 0:
                    Q[i][j] = channel[i][j] * input_prob[i] / output_prob[j]
        
        # Update the input probability distribution
        input_prob = np.exp(np.sum(Q * np.log2(channel), axis=1))
        input_prob /= np.sum(input_prob)
        
        # Compute the new rate
        rate_new = np.sum(input_prob * np.sum(Q * np.log2(Q), axis=1))
    
    return rate_new


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