from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


class Slater:
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, configs):
        r = np.sqrt(np.einsum('ijk,ijk->ij', configs, configs))
        return np.exp(-self.alpha * np.sum(r, axis=1))

    def gradient(self, configs):
        r = np.sqrt(np.einsum('ijk,ijk->ij', configs, configs, optimize=True))
        grad = -self.alpha * (configs / r[:, :, np.newaxis])
        return grad

    def laplacian(self, configs):
        r = np.sqrt(np.einsum('ijk,ijk->ij', configs, configs, optimize=True))
        ndim = configs.shape[2]
        lap = -self.alpha * (ndim / r + self.alpha)
        return np.sum(lap, axis=1)

    def kinetic(self, configs):
        lap = self.laplacian(configs)
        return -0.5 * lap



class Jastrow:
    def __init__(self, beta=1):
        self.beta = beta

    def get_r_vec(self, configs):
        return configs[:, 0, :] - configs[:, 1, :]

    def get_r_ee(self, configs):
        return np.linalg.norm(self.get_r_vec(configs), axis=1)

    def value(self, configs):
        r_ee = self.get_r_ee(configs)
        return np.exp(self.beta * r_ee)

    def gradient(self, configs):
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        grad = np.zeros_like(configs)
        grad[:, 0, :] = self.beta * r_vec / r_ee[:, None]
        grad[:, 1, :] = -self.beta * r_vec / r_ee[:, None]
        return grad

    def laplacian(self, configs):
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        ndim = configs.shape[2]
        lap = np.zeros(configs.shape[0])
        lap[:] = self.beta * (ndim / r_ee - self.beta * np.sum(r_vec**2, axis=1) / r_ee**2)
        return lap



# Background: In quantum mechanics, wave functions describe the quantum state of a system. 
# The Slater and Jastrow wave functions are commonly used in quantum chemistry to model 
# electron interactions. The Slater wave function accounts for the antisymmetry of the 
# wave function due to the Pauli exclusion principle, while the Jastrow factor introduces 
# electron correlation effects. When combining these wave functions, the product of their 
# values, gradients, and Laplacians is used to describe the overall wave function of the 
# system. The gradient and Laplacian of the product of two functions can be derived using 
# the product rule from calculus. The kinetic energy operator in quantum mechanics is 
# related to the Laplacian of the wave function.


class MultiplyWF:
    def __init__(self, wf1, wf2):
        '''Args:
            wf1 (wavefunction object): Slater
            wf2 (wavefunction object): Jastrow           
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
        return lap1 + lap2 + np.sum(grad1 * grad2, axis=(1, 2))

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        return -0.5 * lap


try:
    targets = process_hdf5_to_tuple('30.3', 6)
    target = targets[0]
    np.random.seed(0)
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
                grad_numeric[:, i, d] = (wf_val_shifted - wf_val)/(wf_val*delta)
        rmse = np.sqrt(np.sum((grad_numeric - grad_analytic)**2)/(nconf*nelec*ndim))
        return rmse
    assert np.allclose(test_gradient(
        np.random.randn(5, 2, 3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        1e-4
    ), target)

    target = targets[1]
    np.random.seed(1)
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
                grad_numeric[:, i, d] = (wf_val_shifted - wf_val)/(wf_val*delta)
        rmse = np.sqrt(np.sum((grad_numeric - grad_analytic)**2)/(nconf*nelec*ndim))
        return rmse
    assert np.allclose(test_gradient(
        np.random.randn(5, 2, 3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        1e-5
    ), target)

    target = targets[2]
    np.random.seed(2)
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
                grad_numeric[:, i, d] = (wf_val_shifted - wf_val)/(wf_val*delta)
        rmse = np.sqrt(np.sum((grad_numeric - grad_analytic)**2)/(nconf*nelec*ndim))
        return rmse
    assert np.allclose(test_gradient(
        np.random.randn(5, 2, 3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        1e-6
    ), target)

    target = targets[3]
    np.random.seed(0)
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
                lap_numeric[:, i] += (wf_plus + wf_minus - 2*wf_val)/(wf_val*delta**2)
        return  np.sqrt(np.sum((lap_numeric - lap_analytic)**2)/(nelec*nconf))
    assert np.allclose(test_laplacian(
        np.random.randn(5, 2, 3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        1e-4
    ), target)

    target = targets[4]
    np.random.seed(1)
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
                lap_numeric[:, i] += (wf_plus + wf_minus - 2*wf_val)/(wf_val*delta**2)
        return  np.sqrt(np.sum((lap_numeric - lap_analytic)**2)/(nelec*nconf))
    assert np.allclose(test_laplacian(
        np.random.randn(5, 2, 3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        1e-5
    ), target)

    target = targets[5]
    np.random.seed(2)
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
                lap_numeric[:, i] += (wf_plus + wf_minus - 2*wf_val)/(wf_val*delta**2)
        return  np.sqrt(np.sum((lap_numeric - lap_analytic)**2)/(nelec*nconf))
    assert np.allclose(test_laplacian(
        np.random.randn(5, 2, 3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        1e-6
    ), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e