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




class JastrowWaveFunction:
    def __init__(self, beta=1):
        '''
        Initialize the Jastrow wave function with a parameter beta.
        Args:
            beta: the parameter governing the electron-electron correlation.
        '''
        self.beta = beta

    def compute_electron_vector(self, configs):
        '''Compute the vector between two electrons.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            electron_vector (np.array): (nconfig, ndim)
        '''
        electron_vector = configs[:, 0, :] - configs[:, 1, :]
        return electron_vector

    def compute_electron_separation(self, configs):
        '''Calculate the separation distance between two electrons.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            separation (np.array): (nconfig,)
        '''
        vector = self.compute_electron_vector(configs)
        separation = np.sqrt(np.einsum('ij,ij->i', vector, vector))
        return separation

    def evaluate_wave_function(self, configs):
        '''Evaluate the unnormalized Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            wave_value (np.array): (nconfig,)
        '''
        separation = self.compute_electron_separation(configs)
        wave_value = np.exp(self.beta * separation)
        return wave_value

    def compute_gradient(self, configs):
        '''Compute the gradient of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            gradient (np.array): (nconfig, nelec, ndim)
        '''
        separation = self.compute_electron_separation(configs)
        vector = self.compute_electron_vector(configs)
        gradient = np.zeros_like(configs)

        valid_indices = separation > 0
        grad_coefficient = self.beta / separation[valid_indices, np.newaxis]
        gradient[valid_indices, 0, :] = grad_coefficient * vector[valid_indices]
        gradient[valid_indices, 1, :] = -grad_coefficient * vector[valid_indices]

        return gradient

    def compute_laplacian(self, configs):
        '''Compute the Laplacian of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            laplacian (np.array): (nconfig, nelec)
        '''
        separation = self.compute_electron_separation(configs)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        valid_indices = separation > 0
        laplacian_coefficient = self.beta * (ndim - 1) / separation[valid_indices]
        laplacian[valid_indices, 0] = laplacian_coefficient - self.beta**2
        laplacian[valid_indices, 1] = laplacian_coefficient - self.beta**2

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