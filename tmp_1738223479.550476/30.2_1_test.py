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
        '''
        Initialize the Jastrow factor with a correlation parameter beta.
        Args:
            beta: exponential correlation factor
        '''
        self.beta = beta

    def get_r_vec(self, configs):
        '''Compute the vector r_12 from electron 2 to electron 1.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): vector pointing from r2 to r1, shape (nconf, ndim)
        '''
        return np.subtract(configs[:, 0, :], configs[:, 1, :])

    def get_r_ee(self, configs):
        '''Compute the Euclidean distance between the two electrons.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): distances, shape (nconf,)
        '''
        return np.sqrt(np.sum(self.get_r_vec(configs)**2, axis=1))

    def value(self, configs):
        '''Evaluate the Jastrow factor.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            jast (np.array): Jastrow factor values, shape (nconf,)
        '''
        distances = self.get_r_ee(configs)
        return np.exp(np.multiply(self.beta, distances))

    def gradient(self, configs):
        '''Compute (gradient psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): gradients, shape (nconf, nelec, ndim)
        '''
        vectors = self.get_r_vec(configs)
        distances = self.get_r_ee(configs)[:, np.newaxis]
        grad = np.empty_like(configs)
        grad[:, 0, :] = self.beta * vectors / distances  # Gradient w.r.t. r1
        grad[:, 1, :] = -self.beta * vectors / distances  # Gradient w.r.t. r2
        return grad

    def laplacian(self, configs):
        '''Compute (laplacian psi) / psi.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): Laplacian values, shape (nconf, nelec)
        '''
        vectors = self.get_r_vec(configs)
        distances = self.get_r_ee(configs)
        lap = np.empty((configs.shape[0], 2))
        
        # Calculate Laplacian with respect to r1
        lap[:, 0] = self.beta * (2 / distances - (self.beta) * np.sum(vectors**2, axis=1) / (distances**2))
        
        # Laplacian with respect to r2 (same due to symmetry)
        lap[:, 1] = lap[:, 0]

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