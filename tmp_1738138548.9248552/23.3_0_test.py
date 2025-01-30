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



def mutual_info(channel, prior):
    channel = np.array(channel)
    prior = np.array(prior)

    # Compute the joint probability matrix P(X, Y)
    joint_prob = np.einsum('i,ij->ij', prior, channel)
    
    # Compute the marginal probability P(Y)
    marginal_y = np.sum(joint_prob, axis=0)
    
    # Compute the mutual information using entropy calculations
    entropy_y = -np.sum(marginal_y * np.log2(marginal_y + np.finfo(float).eps))
    entropy_x_given_y = -np.sum(joint_prob * np.log2(joint_prob / marginal_y + np.finfo(float).eps), axis=0)
    expected_entropy_x_given_y = np.sum(entropy_x_given_y * marginal_y)
    
    mutual = entropy_y - expected_entropy_x_given_y
    
    return mutual



# Background: The Blahut-Arimoto algorithm is an iterative method used to compute the channel capacity of a discrete memoryless channel. 
# The channel capacity is the maximum mutual information between the input and output of the channel, optimized over all possible input distributions.
# The algorithm starts with an initial guess for the input distribution and iteratively refines it to maximize the mutual information.
# In each iteration, the algorithm updates the input distribution and computes the mutual information. 
# The process continues until the change in mutual information between iterations is less than a specified error threshold.


def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''
    num_inputs, num_outputs = channel.shape
    
    # Initialize the input distribution uniformly
    p_input = np.full(num_inputs, 1.0 / num_inputs)
    
    # Initialize the rate
    rate_old = 0.0
    
    while True:
        # Compute the conditional distribution of outputs given inputs
        q_output_given_input = channel / np.sum(channel, axis=0, keepdims=True)
        
        # Compute the auxiliary variable q_output
        q_output = np.dot(p_input, channel)
        
        # Update the input distribution
        p_input_new = np.exp(np.sum(q_output_given_input * np.log2(channel / q_output), axis=1))
        p_input_new /= np.sum(p_input_new)
        
        # Compute the new rate (mutual information)
        rate_new = np.sum(p_input_new * np.sum(q_output_given_input * np.log2(channel / q_output), axis=1))
        
        # Check for convergence
        if np.abs(rate_new - rate_old) < e:
            break
        
        # Update the old rate and input distribution
        rate_old = rate_new
        p_input = p_input_new
    
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