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

    def compute_difference_and_distance(self, configs):
        """
        Compute the difference vector and the distance between two electrons.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            diff_vec (np.array): difference vectors, shape (nconf, ndim)
            distance (np.array): distances, shape (nconf,)
        """
        diff_vec = np.subtract(configs[:, 0, :], configs[:, 1, :])
        distance = np.sqrt(np.einsum('ij,ij->i', diff_vec, diff_vec))
        return diff_vec, distance

    def value(self, configs):
        """
        Calculate the Jastrow factor.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            jastrow_values (np.array): Jastrow factor values, shape (nconf,)
        """
        _, distance = self.compute_difference_and_distance(configs)
        return np.exp(self.beta * distance)

    def gradient(self, configs):
        """
        Calculate (gradient psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): gradients, shape (nconf, nelec, ndim)
        """
        diff_vec, distance = self.compute_difference_and_distance(configs)
        grad = np.empty_like(configs)
        factor = self.beta / distance[:, np.newaxis]
        grad[:, 0, :] = factor * diff_vec  # Gradient for electron 1
        grad[:, 1, :] = -factor * diff_vec  # Gradient for electron 2
        return grad

    def laplacian(self, configs):
        """
        Calculate (laplacian psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            laplacian_values (np.array): Laplacian values, shape (nconf, nelec)
        """
        diff_vec, distance = self.compute_difference_and_distance(configs)
        laplacian_values = np.empty((configs.shape[0], 2))

        # Common term for both electrons
        common_term = (2 / distance) - (self.beta * np.sum(diff_vec**2, axis=1) / (distance**2))
        
        laplacian_values[:, 0] = self.beta * common_term
        laplacian_values[:, 1] = self.beta * common_term  # Symmetry in laplacian for both electrons
        
        return laplacian_values


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