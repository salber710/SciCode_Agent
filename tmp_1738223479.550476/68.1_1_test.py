from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




class Slater:
    def __init__(self, alpha):
        '''Args: 
            alpha: exponential decay factor
        '''
        self.alpha = alpha

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        r1_sqr = np.sum(configs[:, 0, :]**2, axis=1)  # Squared distances for electron 1
        r2_sqr = np.sum(configs[:, 1, :]**2, axis=1)  # Squared distances for electron 2
        r1 = np.sqrt(r1_sqr)
        r2 = np.sqrt(r2_sqr)
        val = np.exp(-self.alpha * (r1 + r2))
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r1_sqr = np.sum(configs[:, 0, :]**2, axis=1, keepdims=True)
        r2_sqr = np.sum(configs[:, 1, :]**2, axis=1, keepdims=True)
        grad1 = -self.alpha * configs[:, 0, :] / np.sqrt(r1_sqr)
        grad2 = -self.alpha * configs[:, 1, :] / np.sqrt(r2_sqr)
        grad = np.concatenate([grad1[:, np.newaxis, :], grad2[:, np.newaxis, :]], axis=1)
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r1_sqr = np.sum(configs[:, 0, :]**2, axis=1)
        r2_sqr = np.sum(configs[:, 1, :]**2, axis=1)
        r1 = np.sqrt(r1_sqr)
        r2 = np.sqrt(r2_sqr)
        lap1 = self.alpha**2 - 2 * self.alpha / r1
        lap2 = self.alpha**2 - 2 * self.alpha / r2
        lap = np.column_stack((lap1, lap2))
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        laplacian_values = self.laplacian(configs)
        kin = -0.5 * np.sum(laplacian_values, axis=1)
        return kin


try:
    targets = process_hdf5_to_tuple('68.1', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Slater(alpha=0.5)
    assert cmp_tuple_or_list((wf.value(configs), wf.gradient(configs), wf.laplacian(configs), wf.kinetic(configs)), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Slater(alpha=1)
    assert cmp_tuple_or_list((wf.value(configs), wf.gradient(configs), wf.laplacian(configs), wf.kinetic(configs)), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Slater(alpha=2)
    assert cmp_tuple_or_list((wf.value(configs), wf.gradient(configs), wf.laplacian(configs), wf.kinetic(configs)), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e