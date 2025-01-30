from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: 
# The Slater wave function for a helium atom is a product of exponential functions for each electron, 
# given by psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons 
# from the nucleus. The gradient of the wave function with respect to the electron coordinates is 
# calculated as the derivative of the wave function divided by the wave function itself, resulting in 
# a vector field. The Laplacian is the divergence of the gradient, which in this context is the sum of 
# the second partial derivatives with respect to each spatial dimension. The kinetic energy operator in 
# quantum mechanics is related to the Laplacian by the expression -0.5 * (laplacian psi) / psi.


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
        val = np.exp(-self.alpha * np.sum(r, axis=1))  # Unnormalized wave function
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)  # Shape (nconf, nelec, 1)
        grad = -self.alpha * configs / r  # Gradient of psi divided by psi
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
        lap = -self.alpha * (2 / r + self.alpha)  # Laplacian of psi divided by psi
        return np.sum(lap, axis=2)  # Sum over dimensions

    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)  # (nconf, nelec)
        kin = -0.5 * np.sum(lap, axis=1)  # Kinetic energy per configuration
        return kin


try:
    targets = process_hdf5_to_tuple('30.1', 3)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Slater(alpha=0.5)
    assert cmp_tuple_or_list((wf.value(configs), wf.gradient(configs), wf.laplacian(configs), wf.kinetic(configs)), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Slater(alpha=1)
    assert cmp_tuple_or_list((wf.value(configs), wf.gradient(configs), wf.laplacian(configs), wf.kinetic(configs)), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    configs = np.array([[[ 0.76103773,  0.12167502,  0.44386323], [ 0.33367433,  1.49407907, -0.20515826]]])
    wf = Slater(alpha=2)
    assert cmp_tuple_or_list((wf.value(configs), wf.gradient(configs), wf.laplacian(configs), wf.kinetic(configs)), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e