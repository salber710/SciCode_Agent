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
        Initialize the Jastrow class with a parameter beta.
        Args:
            beta: a parameter that adjusts the electron-electron correlation.
        '''
        self.beta = beta

    def compute_vector(self, configs):
        '''Compute the vector pointing from electron 2 to electron 1.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            vector (np.array): (nconf, ndim)
        '''
        return np.diff(configs, axis=1)[:, 0, :]

    def compute_magnitude(self, configs):
        '''Compute the Euclidean magnitude of the vector from electron 2 to electron 1.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            magnitude (np.array): (nconf,)
        '''
        vector = self.compute_vector(configs)
        return np.sum(np.square(vector), axis=1)**0.5

    def value(self, configs):
        '''Evaluate the unnormalized Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            wave_function (np.array): (nconf,)
        '''
        magnitude = self.compute_magnitude(configs)
        return np.exp(self.beta * magnitude)

    def gradient(self, configs):
        '''Compute the gradient of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            gradient (np.array): (nconf, nelec, ndim)
        '''
        magnitude = self.compute_magnitude(configs)
        vector = self.compute_vector(configs)
        gradient = np.zeros_like(configs)

        nonzero = magnitude != 0
        grad_factor = self.beta / magnitude[nonzero][:, np.newaxis]
        gradient[nonzero, 0, :] = grad_factor * vector[nonzero]
        gradient[nonzero, 1, :] = -grad_factor * vector[nonzero]

        return gradient

    def laplacian(self, configs):
        '''Compute the Laplacian of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            laplacian (np.array): (nconf, nelec)
        '''
        magnitude = self.compute_magnitude(configs)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        nonzero = magnitude != 0
        lap_factor = self.beta * (ndim - 1) / magnitude[nonzero]
        laplacian[nonzero, 0] = lap_factor - self.beta**2
        laplacian[nonzero, 1] = lap_factor - self.beta**2

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