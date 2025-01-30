from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




class Slater:
    def __init__(self, alpha):
        '''Args: 
            alpha: exponential decay factor
        '''
        self.alpha = alpha

    @staticmethod
    def _squared_distance(electron_coords):
        return np.einsum('...i,...i->...', electron_coords, electron_coords)

    def _compute_r_inverse(self, configs):
        r1_sq = self._squared_distance(configs[:, 0, :])
        r2_sq = self._squared_distance(configs[:, 1, :])
        r1_inv = 1 / np.sqrt(r1_sq)
        r2_inv = 1 / np.sqrt(r2_sq)
        return r1_inv, r2_inv

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        r1_inv, r2_inv = self._compute_r_inverse(configs)
        exponent = -self.alpha * (1 / r1_inv + 1 / r2_inv)
        return np.exp(exponent)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r1_inv, r2_inv = self._compute_r_inverse(configs)

        grad_r1 = -self.alpha * configs[:, 0, :] * r1_inv**3
        grad_r2 = -self.alpha * configs[:, 1, :] * r2_inv**3

        return np.stack([grad_r1, grad_r2], axis=1)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r1_inv, r2_inv = self._compute_r_inverse(configs)

        lap_r1 = self.alpha**2 * r1_inv**4 + 2 * self.alpha * r1_inv**3
        lap_r2 = self.alpha**2 * r2_inv**4 + 2 * self.alpha * r2_inv**3

        return np.stack([lap_r1, lap_r2], axis=1)

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        return -0.5 * np.sum(lap, axis=1)


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