from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




class Slater:
    def __init__(self, alpha):
        '''Args: 
            alpha: exponential decay factor
        '''
        self.alpha = alpha

    def _compute_distance_squared(self, coords):
        return np.sum(coords**2, axis=2)

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        r_squared = self._compute_distance_squared(configs)
        r = np.sqrt(r_squared)
        psi = np.exp(-self.alpha * np.sum(r, axis=1))
        return psi

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r_squared = self._compute_distance_squared(configs)[:, :, np.newaxis]
        r = np.sqrt(r_squared)
        grad = -self.alpha * configs / r
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r_squared = self._compute_distance_squared(configs)
        r = np.sqrt(r_squared)
        lap = (self.alpha**2 * r - 2 * self.alpha) / r_squared
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kinetic_energy = -0.5 * np.sum(lap, axis=1)
        return kinetic_energy


try:
    targets = process_hdf5_to_tuple('30.1', 3)
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