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
    channel: a classical channel, 2d array of floats; channel[i][j] means probability of j given i
    prior:   input random variable, 1d array of floats.
    Output
    mutual: mutual information between the input random variable and the random variable associated with the output of the channel, a single scalar value (float)
    '''

    # Calculate the joint distribution p(x, y)
    joint_distribution = np.outer(prior, channel)

    # Calculate the marginal distribution of the output p(y)
    p_y = np.sum(joint_distribution, axis=0)

    # Calculate mutual information I(X;Y)
    mutual = 0.0

    # Iterate over possible input x and output y
    for i in range(len(prior)):
        for j in range(len(p_y)):
            if joint_distribution[i][j] > 0:
                mutual += joint_distribution[i][j] * np.log2(joint_distribution[i][j] / (prior[i] * p_y[j]))

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