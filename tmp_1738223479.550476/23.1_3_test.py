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
    
    # Ensure p and q are lists, if not already
    p = list(p)
    q = list(q)
    
    # Check if distributions are valid
    if not (abs(sum(p) - 1) < 1e-9 and abs(sum(q) - 1) < 1e-9):
        raise ValueError("Input distributions must sum to 1.")
    if any(prob < 0 for prob in p) or any(prob < 0 for prob in q):
        raise ValueError("Input distributions must have non-negative values.")
    
    # Compute KL-divergence
    divergence = 0.0
    for i in range(len(p)):
        if p[i] > 0:
            divergence += p[i] * (math.log(p[i]) - math.log(q[i])) / math.log(2)
    
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