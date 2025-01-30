import numpy as np

# Background: 
# The Slater wave function for a two-electron system like helium is given by the product of exponential functions: 
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons from the nucleus.
# The gradient of the wave function with respect to the electron coordinates is given by the partial derivatives of psi.
# The Laplacian is the sum of the second partial derivatives, which is used to calculate the kinetic energy.
# The kinetic energy operator in quantum mechanics is related to the Laplacian by the expression: 
# T = - (hbar^2 / 2m) * Laplacian(psi) / psi, where hbar is the reduced Planck's constant and m is the electron mass.
# For simplicity, we often use atomic units where hbar = 1 and m = 1/2, so T = -0.5 * Laplacian(psi) / psi.


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
        r = np.linalg.norm(configs, axis=2)  # Calculate the distance of each electron from the origin
        val = np.exp(-self.alpha * r).prod(axis=1)  # Product of exponentials for each configuration
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)  # Shape (nconf, nelec, 1)
        grad = -self.alpha * configs / np.where(r != 0, r, 1)  # Avoid division by zero
        grad = np.where(r != 0, grad, 0)  # Set gradient to zero where distance is zero
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)  # Shape (nconf, nelec, 1)
        ndim = configs.shape[2]
        lap = self.alpha**2 - self.alpha * ndim / np.where(r != 0, r, 1)  # Avoid division by zero
        return lap.sum(axis=1)  # Sum over electrons

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)  # (nconf, nelec)
        kin = -0.5 * lap  # Multiply by -0.5
        return kin


# Background: 
# The Jastrow wave function is a correlation factor used in quantum mechanics to account for electron-electron interactions.
# For a two-electron system, the Jastrow factor is given by psi = exp(beta * |r1 - r2|), where |r1 - r2| is the distance between the two electrons.
# The gradient of the Jastrow wave function with respect to the electron coordinates involves the derivative of the exponential function.
# The Laplacian of the Jastrow wave function involves the second derivatives and is used to calculate the kinetic energy contribution from the Jastrow factor.
# The gradient and Laplacian are calculated with respect to the electron coordinates, and they are divided by the wave function to simplify expressions in quantum Monte Carlo methods.


class Jastrow:
    def __init__(self, beta=1):
        '''
        Args:
            beta: correlation factor
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
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs).reshape(-1, 1)
        grad = np.zeros_like(configs)
        
        # Avoid division by zero
        nonzero_mask = r_ee.flatten() != 0
        grad[nonzero_mask, 0, :] = self.beta * r_vec[nonzero_mask] / r_ee[nonzero_mask]
        grad[nonzero_mask, 1, :] = -self.beta * r_vec[nonzero_mask] / r_ee[nonzero_mask]
        
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf,)        
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs).reshape(-1, 1)
        ndim = configs.shape[2]
        
        lap = np.zeros(configs.shape[0])
        
        # Avoid division by zero
        nonzero_mask = r_ee.flatten() != 0
        lap[nonzero_mask] = 2 * self.beta * (ndim - 1) / r_ee[nonzero_mask].flatten() - 2 * self.beta**2
        
        return lap



# Background: In quantum mechanics, wave functions can be combined to form new wave functions. 
# When two wave functions are multiplied, the resulting wave function represents a system where 
# both wave functions are simultaneously satisfied. The value of the combined wave function is 
# simply the product of the values of the individual wave functions. The gradient of the product 
# of two wave functions is given by the sum of the gradients of the individual wave functions, 
# weighted by their respective values. Similarly, the Laplacian of the product is the sum of the 
# Laplacians of the individual wave functions plus an additional term that accounts for the 
# interaction between the gradients of the two wave functions. The kinetic energy can be derived 
# from the Laplacian, as it is related to the second derivatives of the wave function.


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
            lap (np.array): (nconf,)
        '''
        lap1 = self.wf1.laplacian(configs)
        lap2 = self.wf2.laplacian(configs)
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        
        # Calculate the interaction term: sum over electrons and dimensions
        interaction_term = np.sum(grad1 * grad2, axis=(1, 2))
        
        return lap1 + lap2 + interaction_term

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * lap
        return kin

from scicode.parse.parse import process_hdf5_to_tuple
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
