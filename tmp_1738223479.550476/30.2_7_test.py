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
        """Initialize with the correlation parameter beta."""
        self.beta = beta

    def calculate_distance_and_difference(self, configs):
        """
        Calculate the distance between electrons and the vector difference.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            difference (np.array): vector differences, shape (nconf, ndim)
            distance (np.array): Euclidean distances, shape (nconf,)
        """
        difference = configs[:, 0, :] - configs[:, 1, :]
        distance = np.linalg.norm(difference, axis=1)
        return difference, distance

    def value(self, configs):
        """
        Compute the Jastrow wave function value.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            jastrow (np.array): Jastrow factor values, shape (nconf,)
        """
        _, distance = self.calculate_distance_and_difference(configs)
        return np.exp(self.beta * distance)

    def gradient(self, configs):
        """
        Compute the gradient of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            gradient (np.array): gradients, shape (nconf, nelec, ndim)
        """
        difference, distance = self.calculate_distance_and_difference(configs)
        gradient = np.zeros_like(configs)
        for electron in range(2):
            sign = 1 if electron == 0 else -1
            gradient[:, electron, :] = sign * (self.beta / distance[:, np.newaxis]) * difference
        return gradient

    def laplacian(self, configs):
        """
        Compute the laplacian of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            laplacian (np.array): Laplacian values, shape (nconf, nelec)
        """
        difference, distance = self.calculate_distance_and_difference(configs)
        laplacian = np.zeros((configs.shape[0], 2))
        difference_squared = np.sum(difference**2, axis=1)
        laplacian_term = (2 / distance) - (self.beta * difference_squared / (distance**2))
        for electron in range(2):
            laplacian[:, electron] = self.beta * laplacian_term
        return laplacian


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