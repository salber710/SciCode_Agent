from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


class Slater:
    def __init__(self, alpha):
        '''Args: 
            alpha: exponential decay factor
        '''
        self.alpha = alpha

    def _compute_distances(self, configs):
        return np.sqrt(np.sum(configs**2, axis=-1))

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        distances = self._compute_distances(configs)
        total_distance = np.sum(distances, axis=1)
        val = np.exp(-self.alpha * total_distance)
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        distances = self._compute_distances(configs)
        grad = -self.alpha * configs / distances[:, :, np.newaxis]
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        distances = self._compute_distances(configs)
        lap = (self.alpha**2 - 2 * self.alpha / distances) / distances
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin




class Jastrow:
    def __init__(self, beta=1):
        """
        Initialize the Jastrow factor with a correlation parameter beta.
        Args:
            beta: exponential correlation factor
        """
        self.beta = beta

    def compute_inter_electron_properties(self, configs):
        """
        Compute the vector and Euclidean distance between the two electrons.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): vector from r2 to r1, shape (nconf, ndim)
            r_ee (np.array): distances, shape (nconf,)
        """
        r_vec = np.diff(configs, axis=1)[:, 0, :]
        r_ee = np.sqrt(np.einsum('ij,ij->i', r_vec, r_vec))
        return r_vec, r_ee

    def value(self, configs):
        """
        Calculate the Jastrow factor.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            jast (np.array): Jastrow factor values, shape (nconf,)
        """
        _, r_ee = self.compute_inter_electron_properties(configs)
        return np.exp(self.beta * r_ee)

    def gradient(self, configs):
        """
        Calculate (gradient psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): gradients, shape (nconf, nelec, ndim)
        """
        r_vec, r_ee = self.compute_inter_electron_properties(configs)
        grad = np.zeros_like(configs)
        factor = self.beta / r_ee[:, np.newaxis]
        grad[:, 0, :] = factor * r_vec
        grad[:, 1, :] = -factor * r_vec
        return grad

    def laplacian(self, configs):
        """
        Calculate (laplacian psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): Laplacian values, shape (nconf, nelec)
        """
        r_vec, r_ee = self.compute_inter_electron_properties(configs)
        lap = np.zeros((configs.shape[0], 2))
        squared_dist = r_ee ** 2
        vec_sum_sq = np.einsum('ij,ij->i', r_vec, r_vec)
        common_term = 2 / r_ee - self.beta * vec_sum_sq / squared_dist
        
        lap[:, 0] = self.beta * common_term
        lap[:, 1] = self.beta * common_term
        return lap


try:
    targets = process_hdf5_to_tuple('30.2', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Jastrow(beta=0.5)
    assert cmp_tuple_or_list((wf.get_r_vec(configs), wf.get_r_ee(configs), wf.value(configs), wf.gradient(configs), wf.laplacian(configs)), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Jastrow(beta=1)
    assert cmp_tuple_or_list((wf.get_r_vec(configs), wf.get_r_ee(configs), wf.value(configs), wf.gradient(configs), wf.laplacian(configs)), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Jastrow(beta=2)
    assert cmp_tuple_or_list((wf.get_r_vec(configs), wf.get_r_ee(configs), wf.value(configs), wf.gradient(configs), wf.laplacian(configs)), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e