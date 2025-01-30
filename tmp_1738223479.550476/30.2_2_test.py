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

    def calculate_distance_and_vector(self, configs):
        """
        Computes the distance and vector between two electrons in each configuration.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): Euclidean distances, shape (nconf,)
            r_vec (np.array): vectors pointing from r2 to r1, shape (nconf, ndim)
        """
        r_vec = configs[:, 0, :] - configs[:, 1, :]
        r_ee = np.einsum('ij,ij->i', r_vec, r_vec) ** 0.5
        return r_ee, r_vec

    def value(self, configs):
        """
        Calculate the Jastrow factor.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            jast (np.array): Jastrow factor values, shape (nconf,)
        """
        r_ee, _ = self.calculate_distance_and_vector(configs)
        return np.exp(self.beta * r_ee)

    def gradient(self, configs):
        """
        Calculate (gradient psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): gradients, shape (nconf, nelec, ndim)
        """
        r_ee, r_vec = self.calculate_distance_and_vector(configs)
        grad = np.zeros_like(configs)
        for i in range(configs.shape[1]):  # loop over electrons
            sign = 1 if i == 0 else -1
            grad[:, i, :] = sign * self.beta * r_vec / r_ee[:, np.newaxis]
        return grad

    def laplacian(self, configs):
        """
        Calculate (laplacian psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): Laplacian values, shape (nconf, nelec)
        """
        r_ee, r_vec = self.calculate_distance_and_vector(configs)
        lap = np.zeros((configs.shape[0], 2))
        for i in range(2):  # loop over electrons
            lap[:, i] = self.beta * ((2 / r_ee) - (self.beta * np.einsum('ij,ij->i', r_vec, r_vec) / (r_ee ** 2)))
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