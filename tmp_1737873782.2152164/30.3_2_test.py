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
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)
        return np.exp(-self.alpha * r1) * np.exp(-self.alpha * r2)

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)

        grad1 = -self.alpha * configs[:, 0, :] / r1[:, np.newaxis]
        grad2 = -self.alpha * configs[:, 1, :] / r2[:, np.newaxis]

        return np.stack((grad1, grad2), axis=1)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)

        lap1 = self.alpha * (2.0 / r1 - self.alpha)
        lap2 = self.alpha * (2.0 / r2 - self.alpha)

        return np.stack((lap1, lap2), axis=1)

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
        Args:
            beta: exponential factor for the Jastrow wave function
        '''
        self.beta = beta

    def get_r_vec(self, configs):
        '''Returns a vector pointing from r2 to r1, which is r_12 = [x1 - x2, y1 - y2, z1 - z2].
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): (nconf, ndim)
        '''
        r_vec = configs[:, 0, :] - configs[:, 1, :]
        return r_vec

    def get_r_ee(self, configs):
        '''Returns the Euclidean distance from r2 to r1
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): (nconf,)
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = np.linalg.norm(r_vec, axis=1)
        return r_ee

    def value(self, configs):
        '''Calculate Jastrow factor
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns 
            jast (np.array): (nconf,)
        '''
        r_ee = self.get_r_ee(configs)
        jast = np.exp(self.beta * r_ee)
        return jast

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r_ee = self.get_r_ee(configs)
        r_vec = self.get_r_vec(configs)

        grad1 = self.beta * r_vec / r_ee[:, np.newaxis]
        grad2 = -grad1

        return np.stack((grad1, grad2), axis=1)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_ee = self.get_r_ee(configs)
        lap1 = self.beta * (2.0 / r_ee + self.beta)
        lap2 = lap1

        return np.stack((lap1, lap2), axis=1)




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

        # laplacian of product is: (lap1 + lap2 + 2 * dot(grad1, grad2))
        lap = lap1 + lap2 + 2 * np.sum(grad1 * grad2, axis=2)
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin


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