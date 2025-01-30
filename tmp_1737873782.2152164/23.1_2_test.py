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
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Make sure both distributions are normalized
    p /= np.sum(p)
    q /= np.sum(q)

    # Calculate the KL divergence using log base 2
    # Use np.where to handle cases where p or q is 0, to avoid log(0) or division by zero
    divergence = np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    return divergence


try:
    targets = process_hdf5_to_tuple('23.1', 3)
    target = targets[0]
    p = [1/2,1/2,0,0]
    q = [1/4,1/4,1/4,1/4]
    assert np.allclose(KL_divergence(p,q), target)

    target = targets[1]
    p = [1/2,1/2]
    q = [1/2,1/2]
    assert np.allclose(KL_divergence(p,q), target)

    target = targets[2]
    p = [1,0]
    q = [1/4,3/4]
    assert np.allclose(KL_divergence(p,q), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e