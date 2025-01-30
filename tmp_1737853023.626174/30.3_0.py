import numpy as np

# Background: 
# The Slater wave function for a two-electron system like helium is given by the product of exponential functions: 
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons from the nucleus.
# The gradient of the wave function with respect to the electron coordinates is given by the partial derivatives of psi.
# The Laplacian is the sum of the second partial derivatives, which is used to calculate the kinetic energy.
# The kinetic energy operator in quantum mechanics is related to the Laplacian by the expression: 
# T = - (hbar^2 / 2m) * Laplacian(psi), where hbar is the reduced Planck's constant and m is the electron mass.
# For simplicity, we often work in atomic units where hbar = 1 and m = 1/2, so T = -0.5 * Laplacian(psi).


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
        val = np.exp(-self.alpha * (r1 + r2))
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r1 = np.linalg.norm(configs[:, 0, :], axis=1, keepdims=True)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1, keepdims=True)
        grad1 = -self.alpha * configs[:, 0, :] / np.where(r1 == 0, 1, r1)
        grad2 = -self.alpha * configs[:, 1, :] / np.where(r2 == 0, 1, r2)
        grad = np.stack((grad1, grad2), axis=1)
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)
        lap1 = self.alpha**2 - 2 * self.alpha / np.where(r1 == 0, 1, r1)
        lap2 = self.alpha**2 - 2 * self.alpha / np.where(r2 == 0, 1, r2)
        lap = np.stack((lap1, lap2), axis=1)
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


# Background: 
# The Jastrow wave function is a correlation factor used in quantum mechanics to account for electron-electron interactions.
# For a two-electron system, the Jastrow factor is given by psi = exp(beta * |r1 - r2|), where |r1 - r2| is the distance between the two electrons.
# The gradient of the Jastrow wave function with respect to the electron coordinates involves the derivative of the exponential function.
# The Laplacian of the Jastrow wave function involves the second derivatives and is used to calculate the kinetic energy contribution from the Jastrow factor.
# The gradient and Laplacian are calculated with respect to the electron coordinates, and the results are normalized by dividing by the wave function itself.


class Jastrow:
    def __init__(self, beta=1):
        '''Initialize the Jastrow factor with a given beta parameter.'''
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
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        grad_factor = self.beta / np.where(r_ee == 0, 1, r_ee)[:, np.newaxis]
        grad1 = grad_factor[:, np.newaxis] * r_vec
        grad2 = -grad1
        grad = np.stack((grad1, grad2), axis=1)
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        ndim = configs.shape[2]
        
        # Calculate the Laplacian
        lap_factor = self.beta / np.where(r_ee == 0, 1, r_ee)
        lap1 = (self.beta**2) * np.ones_like(r_ee) - lap_factor * ndim
        lap2 = lap1
        
        lap = np.stack((lap1, lap2), axis=1)
        return lap



# Background: 
# In quantum mechanics, wave functions can be combined to form new wave functions. 
# When two wave functions are multiplied, the resulting wave function's value is the product of the individual wave function values.
# The gradient of the product of two wave functions is given by the sum of the gradients of the individual wave functions, each weighted by the value of the other wave function.
# Mathematically, if psi = psi1 * psi2, then the gradient of psi is (grad psi1 / psi1 + grad psi2 / psi2).
# The Laplacian of the product of two wave functions involves the Laplacians of the individual wave functions and their gradients.
# Specifically, (laplacian psi) / psi = (laplacian psi1) / psi1 + (laplacian psi2) / psi2 + 2 * (grad psi1 / psi1) * (grad psi2 / psi2).
# The kinetic energy operator is related to the Laplacian, and for the product wave function, it can be calculated using the combined Laplacian.


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
        val = val1 * val2
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        grad = grad1 + grad2
        return grad

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
        cross_term = 2 * np.sum(grad1 * grad2, axis=2)
        lap = lap1 + lap2 + cross_term[:, np.newaxis]
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

from scicode.parse.parse import process_hdf5_to_tuple
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
