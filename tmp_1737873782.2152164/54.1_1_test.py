from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



def basis(i, p, M, h, etype):
    '''Inputs
    i: int, the index of element
    p: array of arbitrary size 1,2, or 3, the coordinates
    M: int, the total number of the nodal dofs
    h: int, the element size
    etype: int, basis function type; When type equals to 1, 
    it returns $\omega^1(x)$, when the type equals to 2, it returns the value of function $\omega^2(x)$.
    Outputs
    v: array of size 1,2, or 3, value of basis function
    '''
    
    # Initialize the output array
    v = np.zeros_like(p)
    
    # Determine the shape function based on the etype
    if etype == 1:
        # Calculate omega^1(x)
        for idx, x in enumerate(p):
            if x >= i * h and x <= (i + 1) * h:
                v[idx] = (x - i * h) / h
            else:
                v[idx] = 0
    
    elif etype == 2:
        # Calculate omega^2(x)
        for idx, x in enumerate(p):
            if x >= (i - 1) * h and x <= i * h:
                v[idx] = ((i + 1) * h - x) / h
            else:
                v[idx] = 0
    
    return v


try:
    targets = process_hdf5_to_tuple('54.1', 3)
    target = targets[0]
    i = 1
    p = np.array([0.2])
    M = 10 
    h = 0.1 
    etype = 1
    assert np.allclose(basis(i, p, M, h, etype), target)

    target = targets[1]
    i = 2
    p = np.array([0.5,0.1])
    M = 20 
    h = 0.5 
    etype = 1
    assert np.allclose(basis(i, p, M, h, etype), target)

    target = targets[2]
    i = 3
    p = np.array([5,7,9])
    M = 30 
    h = 0.01 
    etype = 2
    assert np.allclose(basis(i, p, M, h, etype), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e