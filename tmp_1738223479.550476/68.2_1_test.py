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
        r = np.sqrt(np.sum(configs**2, axis=2))
        val = np.exp(-self.alpha * np.sum(r, axis=1))
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.sqrt(np.sum(configs**2, axis=2))
        grad = np.full(configs.shape, -self.alpha)
        r_nonzero = r != 0
        grad[r_nonzero] *= configs[r_nonzero] / r[r_nonzero][:, :, np.newaxis]
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.sqrt(np.sum(configs**2, axis=2))
        lap = np.full(r.shape, self.alpha**2)
        r_nonzero = r != 0
        lap[r_nonzero] -= 2 * self.alpha / r[r_nonzero]
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy
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
        Initialize the Jastrow class with a correlation strength parameter beta.
        Args:
            beta: a parameter that controls the strength of the electron-electron correlation.
        '''
        self.beta = beta

    def compute_distance_vector(self, configs):
        '''Calculate the vector from electron 2 to electron 1.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            distance_vector (np.array): (nconf, ndim)
        '''
        return configs[:, 0, :] - configs[:, 1, :]

    def compute_distance(self, configs):
        '''Calculate the Euclidean distance between electron 1 and electron 2.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            distance (np.array): (nconf,)
        '''
        distance_vector = self.compute_distance_vector(configs)
        return np.linalg.norm(distance_vector, axis=1)

    def value(self, configs):
        '''Evaluate the Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            jastrow_value (np.array): (nconf,)
        '''
        distance = self.compute_distance(configs)
        return np.exp(self.beta * distance)

    def gradient(self, configs):
        '''Calculate the gradient of the Jastrow wave function divided by the Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            gradient (np.array): (nconf, nelec, ndim)
        '''
        distance = self.compute_distance(configs)
        distance_vector = self.compute_distance_vector(configs)
        gradient = np.zeros_like(configs)

        nonzero_mask = distance > 0 
        grad_factor = self.beta / distance[nonzero_mask, np.newaxis]
        gradient[nonzero_mask, 0, :] = grad_factor * distance_vector[nonzero_mask]
        gradient[nonzero_mask, 1, :] = -grad_factor * distance_vector[nonzero_mask]

        return gradient

    def laplacian(self, configs):
        '''Calculate the Laplacian of the Jastrow wave function divided by the Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            laplacian (np.array): (nconf, nelec)
        '''
        distance = self.compute_distance(configs)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        nonzero_mask = distance > 0
        laplacian_term = self.beta * (ndim - 1) / distance[nonzero_mask]
        laplacian[nonzero_mask, 0] = laplacian_term - self.beta**2
        laplacian[nonzero_mask, 1] = laplacian_term - self.beta**2

        return laplacian


try:
    targets = process_hdf5_to_tuple('68.2', 3)
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