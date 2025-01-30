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