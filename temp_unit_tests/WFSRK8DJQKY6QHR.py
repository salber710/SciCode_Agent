
# Background: The Blahut-Arimoto algorithm is an iterative method used to compute the channel capacity of a discrete memoryless channel. The channel capacity is the maximum mutual information between the input and output of the channel, optimized over all possible input distributions. The algorithm starts with an initial guess for the input distribution and iteratively refines it to maximize the mutual information. In each iteration, the algorithm updates the input distribution and computes the mutual information. The process continues until the change in mutual information between iterations is less than a specified error threshold, indicating convergence.

import numpy as np

def blahut_arimoto(channel, e):
    '''Input
    channel: a classical channel, 2d array of floats; Channel[i][j] means probability of i given j
    e:       error threshold, a single scalar value (float)
    Output
    rate_new: channel capacity, a single scalar value (float)
    '''
    num_inputs, num_outputs = channel.shape
    
    # Initialize the input distribution uniformly
    p_x = np.full(num_inputs, 1.0 / num_inputs)
    
    # Initialize the rate
    rate_old = 0.0
    
    while True:
        # Compute the conditional distribution P(y|x) * P(x)
        p_y_given_x_times_p_x = channel * p_x[:, np.newaxis]
        
        # Compute the marginal distribution P(y)
        p_y = np.sum(p_y_given_x_times_p_x, axis=0)
        
        # Compute the updated input distribution
        q_x = np.zeros(num_inputs)
        for i in range(num_inputs):
            for j in range(num_outputs):
                if channel[i, j] > 0:
                    q_x[i] += channel[i, j] * np.log2(channel[i, j] / p_y[j])
            q_x[i] = np.exp(q_x[i])
        
        # Normalize the updated input distribution
        q_x /= np.sum(q_x)
        
        # Compute the new rate (mutual information)
        rate_new = 0.0
        for i in range(num_inputs):
            for j in range(num_outputs):
                if channel[i, j] > 0:
                    rate_new += p_x[i] * channel[i, j] * np.log2(channel[i, j] / p_y[j])
        
        # Check for convergence
        if abs(rate_new - rate_old) < e:
            break
        
        # Update the input distribution and rate for the next iteration
        p_x = q_x
        rate_old = rate_new
    
    return rate_new
