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
            beta: a parameter controlling the electron-electron correlation strength.
        '''
        self.beta = beta

    def electron_vector_difference(self, configs):
        '''Calculate the vector pointing from electron 2 to electron 1.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            vector_diff (np.array): (nconfig, ndim)
        '''
        vector_diff = np.subtract(configs[:, 0, :], configs[:, 1, :])
        return vector_diff

    def electron_distance(self, vector_diff):
        '''Calculate the Euclidean distance using the vector difference.
        Args:
            vector_diff (np.array): vector differences of shape (nconfig, ndim)
        Returns:
            distances (np.array): (nconfig,)
        '''
        distances = np.sqrt(np.sum(vector_diff**2, axis=1))
        return distances

    def value(self, configs):
        '''Evaluate the Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            jastrow_value (np.array): (nconfig,)
        '''
        vector_diff = self.electron_vector_difference(configs)
        distances = self.electron_distance(vector_diff)
        jastrow_value = np.exp(self.beta * distances)
        return jastrow_value

    def gradient(self, configs):
        '''Compute the gradient of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            gradient (np.array): (nconfig, nelec, ndim)
        '''
        vector_diff = self.electron_vector_difference(configs)
        distances = self.electron_distance(vector_diff)
        gradient = np.zeros_like(configs)

        mask = distances > 0
        grad_coeff = self.beta / distances[mask, np.newaxis]
        gradient[mask, 0, :] = grad_coeff * vector_diff[mask]
        gradient[mask, 1, :] = -grad_coeff * vector_diff[mask]

        return gradient

    def laplacian(self, configs):
        '''Compute the Laplacian of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            laplacian (np.array): (nconfig, nelec)
        '''
        vector_diff = self.electron_vector_difference(configs)
        distances = self.electron_distance(vector_diff)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        mask = distances > 0
        lap_coefficient = self.beta * (ndim - 1) / distances[mask]
        laplacian[mask, 0] = lap_coefficient - self.beta**2
        laplacian[mask, 1] = lap_coefficient - self.beta**2

        return laplacian



# Background: In quantum mechanics, wave functions describe the quantum state of a system. 
# For systems with multiple interacting particles, wave functions can be complex and may involve products 
# of simpler functions to account for different types of interactions. When considering two wave 
# functions, such as the Slater and Jastrow functions, their product forms a new wave function that 
# can describe a more complex system. The properties of this new wave function can be determined by 
# calculating its value, gradient, and laplacian. For the product of two wave functions, the value is 
# simply the product of the values of the two wave functions. The gradient and laplacian of the product 
# involve the gradients and laplacians of the individual wave functions, which is akin to applying the 
# product rule from calculus.


class MultiplyWF:
    def __init__(self, wf1, wf2):
        '''Args:
            wf1 (wavefunction object): 
            wf2 (wavefunction object):            
        '''
        self.wf1 = wf1
        self.wf2 = wf2
    
    def value(self, configs):
        '''Multiply two wave function values
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        val1 = self.wf1.value(configs)
        val2 = self.wf2.value(configs)
        return val1 * val2
    
    def gradient(self, configs):
        '''Calculate (gradient psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        return grad1 + grad2
    
    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        lap1 = self.wf1.laplacian(configs)
        lap2 = self.wf2.laplacian(configs)
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        # Using the identity: (laplacian psi1 * psi2) / (psi1 * psi2) = (lap psi1) / psi1 + (lap psi2) / psi2 + 2 * (grad psi1 / psi1) \cdot (grad psi2 / psi2)
        lap_combined = lap1 + lap2 + 2 * np.sum(grad1 * grad2, axis=(1, 2))
        return lap_combined

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin


try:
    targets = process_hdf5_to_tuple('68.3', 6)
    target = targets[0]
    def test_gradient(configs, wf, delta):
        '''
        Calculate RMSE between numerical and analytic gradients.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
            wf (wavefunction object):
            delta (float): small move in one dimension
        Returns:
            rmse (float): should be a small number
        '''
        nconf, nelec, ndim = configs.shape
        wf_val = wf.value(configs)
        grad_analytic = wf.gradient(configs)
        grad_numeric = np.zeros(grad_analytic.shape)
        for i in range(nelec):
            for d in range(ndim):
                shift = np.zeros(configs.shape)
                shift[:, i, d] += delta
                wf_val_shifted = wf.value(configs + shift)
                grad_numeric[:, i, d] = (wf_val_shifted - wf_val) / (wf_val * delta)
        rmse = np.sqrt(np.sum((grad_numeric - grad_analytic) ** 2) / (nconf * nelec * ndim))
        return rmse
    np.random.seed(0)
    assert np.allclose(test_gradient(
        np.random.randn(5, 2, 3),
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)),
        1e-4
    ), target)

    target = targets[1]
    def test_gradient(configs, wf, delta):
        '''
        Calculate RMSE between numerical and analytic gradients.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
            wf (wavefunction object):
            delta (float): small move in one dimension
        Returns:
            rmse (float): should be a small number
        '''
        nconf, nelec, ndim = configs.shape
        wf_val = wf.value(configs)
        grad_analytic = wf.gradient(configs)
        grad_numeric = np.zeros(grad_analytic.shape)
        for i in range(nelec):
            for d in range(ndim):
                shift = np.zeros(configs.shape)
                shift[:, i, d] += delta
                wf_val_shifted = wf.value(configs + shift)
                grad_numeric[:, i, d] = (wf_val_shifted - wf_val) / (wf_val * delta)
        rmse = np.sqrt(np.sum((grad_numeric - grad_analytic) ** 2) / (nconf * nelec * ndim))
        return rmse
    np.random.seed(1)
    assert np.allclose(test_gradient(
        np.random.randn(5, 2, 3),
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)),
        1e-5
    ), target)

    target = targets[2]
    def test_gradient(configs, wf, delta):
        '''
        Calculate RMSE between numerical and analytic gradients.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
            wf (wavefunction object):
            delta (float): small move in one dimension
        Returns:
            rmse (float): should be a small number
        '''
        nconf, nelec, ndim = configs.shape
        wf_val = wf.value(configs)
        grad_analytic = wf.gradient(configs)
        grad_numeric = np.zeros(grad_analytic.shape)
        for i in range(nelec):
            for d in range(ndim):
                shift = np.zeros(configs.shape)
                shift[:, i, d] += delta
                wf_val_shifted = wf.value(configs + shift)
                grad_numeric[:, i, d] = (wf_val_shifted - wf_val) / (wf_val * delta)
        rmse = np.sqrt(np.sum((grad_numeric - grad_analytic) ** 2) / (nconf * nelec * ndim))
        return rmse
    np.random.seed(2)
    assert np.allclose(test_gradient(
        np.random.randn(5, 2, 3),
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)),
        1e-6
    ), target)

    target = targets[3]
    def test_laplacian(configs, wf, delta=1e-5):
        '''
        Calculate RMSE between numerical and analytic laplacians.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
            wf (wavefunction object):
            delta (float): small move in one dimension
        Returns:
            rmse (float): should be a small number
        '''
        nconf, nelec, ndim = configs.shape
        wf_val = wf.value(configs)
        lap_analytic = wf.laplacian(configs)
        lap_numeric = np.zeros(lap_analytic.shape)
        for i in range(nelec):
            for d in range(ndim):
                shift = np.zeros(configs.shape)
                shift_plus = shift.copy()
                shift_plus[:, i, d] += delta
                wf_plus = wf.value(configs + shift_plus)
                shift_minus = shift.copy()
                shift_minus[:, i, d] -= delta
                wf_minus = wf.value(configs + shift_minus)
                lap_numeric[:, i] += (wf_plus + wf_minus - 2 * wf_val) / (wf_val * delta ** 2)
        return np.sqrt(np.sum((lap_numeric - lap_analytic) ** 2) / (nelec * nconf))
    np.random.seed(0)
    assert np.allclose(test_laplacian(
        np.random.randn(5, 2, 3),
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)),
        1e-4
    ), target)

    target = targets[4]
    def test_laplacian(configs, wf, delta=1e-5):
        '''
        Calculate RMSE between numerical and analytic laplacians.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
            wf (wavefunction object):
            delta (float): small move in one dimension
        Returns:
            rmse (float): should be a small number
        '''
        nconf, nelec, ndim = configs.shape
        wf_val = wf.value(configs)
        lap_analytic = wf.laplacian(configs)
        lap_numeric = np.zeros(lap_analytic.shape)
        for i in range(nelec):
            for d in range(ndim):
                shift = np.zeros(configs.shape)
                shift_plus = shift.copy()
                shift_plus[:, i, d] += delta
                wf_plus = wf.value(configs + shift_plus)
                shift_minus = shift.copy()
                shift_minus[:, i, d] -= delta
                wf_minus = wf.value(configs + shift_minus)
                lap_numeric[:, i] += (wf_plus + wf_minus - 2 * wf_val) / (wf_val * delta ** 2)
        return np.sqrt(np.sum((lap_numeric - lap_analytic) ** 2) / (nelec * nconf))
    np.random.seed(1)
    assert np.allclose(test_laplacian(
        np.random.randn(5, 2, 3),
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)),
        1e-5
    ), target)

    target = targets[5]
    def test_laplacian(configs, wf, delta=1e-5):
        '''
        Calculate RMSE between numerical and analytic laplacians.
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
            wf (wavefunction object):
            delta (float): small move in one dimension
        Returns:
            rmse (float): should be a small number
        '''
        nconf, nelec, ndim = configs.shape
        wf_val = wf.value(configs)
        lap_analytic = wf.laplacian(configs)
        lap_numeric = np.zeros(lap_analytic.shape)
        for i in range(nelec):
            for d in range(ndim):
                shift = np.zeros(configs.shape)
                shift_plus = shift.copy()
                shift_plus[:, i, d] += delta
                wf_plus = wf.value(configs + shift_plus)
                shift_minus = shift.copy()
                shift_minus[:, i, d] -= delta
                wf_minus = wf.value(configs + shift_minus)
                lap_numeric[:, i] += (wf_plus + wf_minus - 2 * wf_val) / (wf_val * delta ** 2)
        return np.sqrt(np.sum((lap_numeric - lap_analytic) ** 2) / (nelec * nconf))
    np.random.seed(2)
    assert np.allclose(test_laplacian(
        np.random.randn(5, 2, 3),
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)),
        1e-6
    ), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e