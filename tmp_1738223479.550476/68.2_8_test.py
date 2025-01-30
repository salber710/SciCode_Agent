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
        Initializes the Jastrow class with a correlation parameter beta.
        Args:
            beta: A parameter that dictates the electron-electron interaction strength.
        '''
        self.beta = beta

    def calculate_distance_vector(self, configs):
        '''Finds the vector difference from electron 1 to electron 2.
        Args:
            configs (np.array): Electron positions of shape (nconfig, nelec, ndim)
        Returns:
            distance_vector (np.array): (nconfig, ndim)
        '''
        return configs[:, 1, :] - configs[:, 0, :]

    def calculate_distance_magnitude(self, configs):
        '''Computes the Euclidean distance between two electrons.
        Args:
            configs (np.array): Electron positions of shape (nconfig, nelec, ndim)
        Returns:
            distance_magnitude (np.array): (nconfig,)
        '''
        distance_vector = self.calculate_distance_vector(configs)
        return np.sqrt(np.einsum('ij,ij->i', distance_vector, distance_vector))

    def value(self, configs):
        '''Computes the Jastrow wave function value.
        Args:
            configs (np.array): Electron positions of shape (nconfig, nelec, ndim)
        Returns:
            wave_function_value (np.array): (nconfig,)
        '''
        distance_magnitude = self.calculate_distance_magnitude(configs)
        return np.exp(self.beta * distance_magnitude)

    def gradient(self, configs):
        '''Calculates the gradient of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): Electron positions of shape (nconfig, nelec, ndim)
        Returns:
            gradient (np.array): (nconfig, nelec, ndim)
        '''
        distance_magnitude = self.calculate_distance_magnitude(configs)
        distance_vector = self.calculate_distance_vector(configs)
        gradient = np.zeros_like(configs)

        valid_indices = distance_magnitude > 0
        grad_factor = self.beta / distance_magnitude[valid_indices, np.newaxis]
        gradient[valid_indices, 1, :] = grad_factor * distance_vector[valid_indices]
        gradient[valid_indices, 0, :] = -grad_factor * distance_vector[valid_indices]

        return gradient

    def laplacian(self, configs):
        '''Computes the Laplacian of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): Electron positions of shape (nconfig, nelec, ndim)
        Returns:
            laplacian (np.array): (nconfig, nelec)
        '''
        distance_magnitude = self.calculate_distance_magnitude(configs)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        valid_indices = distance_magnitude > 0
        laplace_term = self.beta * (ndim - 1) / distance_magnitude[valid_indices]
        laplacian[valid_indices, 1] = laplace_term - self.beta**2
        laplacian[valid_indices, 0] = laplace_term - self.beta**2

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