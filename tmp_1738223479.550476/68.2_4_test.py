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




class JastrowFunction:
    def __init__(self, beta=1):
        '''
        Initialize the JastrowFunction with a correlation parameter beta.
        Args:
            beta: parameter controlling the correlation strength.
        '''
        self.beta = beta

    def relative_vector(self, configs):
        '''Compute the vector difference between the two electrons for each configuration.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            rel_vec (np.array): (nconfig, ndim)
        '''
        return np.subtract(configs[:, 0, :], configs[:, 1, :])

    def electron_distance(self, configs):
        '''Calculate the electron-electron distance for each configuration.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            distance (np.array): (nconfig,)
        '''
        vector = self.relative_vector(configs)
        return np.sum(vector**2, axis=1)**0.5

    def evaluate(self, configs):
        '''Evaluate the unnormalized wave function for each configuration.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            wave_fn (np.array): (nconfig,)
        '''
        distance = self.electron_distance(configs)
        return np.exp(self.beta * distance)

    def calc_gradient(self, configs):
        '''Calculate the gradient of the wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            gradient (np.array): (nconfig, nelec, ndim)
        '''
        distance = self.electron_distance(configs)
        vec = self.relative_vector(configs)
        gradient = np.zeros_like(configs)

        non_zero_dist = distance > 0
        grad_coeff = self.beta / distance[non_zero_dist, np.newaxis]
        gradient[non_zero_dist, 0, :] = grad_coeff * vec[non_zero_dist]
        gradient[non_zero_dist, 1, :] = -grad_coeff * vec[non_zero_dist]

        return gradient

    def calc_laplacian(self, configs):
        '''Calculate the Laplacian of the wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            laplacian (np.array): (nconfig, nelec)
        '''
        distance = self.electron_distance(configs)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        non_zero_dist = distance > 0
        laplacian_coef = self.beta * (ndim - 1) / distance[non_zero_dist]
        laplacian[non_zero_dist, 0] = laplacian_coef - self.beta**2
        laplacian[non_zero_dist, 1] = laplacian_coef - self.beta**2

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