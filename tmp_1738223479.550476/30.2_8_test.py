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
        """Initialize the Jastrow wave function with a correlation parameter beta."""
        self.beta = beta

    def compute_vector_and_magnitude(self, configs):
        """
        Calculate the vector and magnitude between two electrons.
        Args:
            configs (np.array): Electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            vector (np.array): Vectors from r2 to r1, shape (nconf, ndim)
            magnitude (np.array): Magnitudes, shape (nconf,)
        """
        vector = configs[:, 0, :] - configs[:, 1, :]
        magnitude = np.linalg.norm(vector, axis=-1)
        return vector, magnitude

    def value(self, configs):
        """
        Evaluate the Jastrow factor.
        Args:
            configs (np.array): Electron configurations, shape (nconf, nelec, ndim)
        Returns:
            jastrow_values (np.array): Jastrow factor values, shape (nconf,)
        """
        _, magnitude = self.compute_vector_and_magnitude(configs)
        return np.power(np.e, self.beta * magnitude)

    def gradient(self, configs):
        """
        Compute (gradient psi) / psi.
        Args:
            configs (np.array): Electron configurations, shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): Gradient values, shape (nconf, nelec, ndim)
        """
        vector, magnitude = self.compute_vector_and_magnitude(configs)
        grad = np.zeros_like(configs)
        grad[:, 0, :] = (self.beta * vector / magnitude[:, np.newaxis])
        grad[:, 1, :] = -(self.beta * vector / magnitude[:, np.newaxis])
        return grad

    def laplacian(self, configs):
        """
        Compute (laplacian psi) / psi.
        Args:
            configs (np.array): Electron configurations, shape (nconf, nelec, ndim)
        Returns:
            laplacian (np.array): Laplacian values, shape (nconf, nelec)
        """
        vector, magnitude = self.compute_vector_and_magnitude(configs)
        laplacian = np.zeros((configs.shape[0], 2))
        common_term = (2 / magnitude) - (self.beta * np.sum(vector**2, axis=1) / (magnitude**2))
        laplacian[:, 0] = self.beta * common_term
        laplacian[:, 1] = self.beta * common_term
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