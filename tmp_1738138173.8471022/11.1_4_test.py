from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools
import scipy.linalg



def ket(dim, args):


    def basis_vec(d, idx):
        return np.eye(d)[idx]

    if isinstance(args, list):
        result = basis_vec(dim[0], args[0])
        for d, idx in zip(dim[1:], args[1:]):
            result = np.kron(result, basis_vec(d, idx))
    else:
        result = basis_vec(dim, args)

    return result


try:
    targets = process_hdf5_to_tuple('11.1', 3)
    target = targets[0]
    assert np.allclose(ket(2, 0), target)

    target = targets[1]
    assert np.allclose(ket(2, [1,1]), target)

    target = targets[2]
    assert np.allclose(ket([2,3], [0,1]), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e