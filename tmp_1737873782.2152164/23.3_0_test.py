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
    # Number of input and output symbols
    m, n = channel.shape
    
    # Initialize input distribution uniformly
    Q = np.ones(m) / m
    
    # Initialize the rate
    rate_old = 0
    rate_new = 0
    
    # Iterate until convergence
    while True:
        # Update the output distribution P(y)
        P_y = np.dot(Q, channel)
        
        # Initialize the conditional distribution and the new rate
        W = np.zeros((m, n))
        rate_new = 0
        
        # Update W(i|j) and calculate the new rate
        for i in range(m):
            for j in range(n):
                if channel[i][j] > 0:
                    W[i][j] = Q[i] * channel[i][j] / P_y[j]
                    rate_new += Q[i] * channel[i][j] * np.log2(W[i][j] / Q[i])
        
        # Update the input distribution Q(i)
        for i in range(m):
            Q[i] = np.sum(W[i])
        
        # Check for convergence
        if np.abs(rate_new - rate_old) < e:
            break
        
        # Update the old rate
        rate_old = rate_new
    
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