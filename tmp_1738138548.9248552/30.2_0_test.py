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



# Background: 
# The Jastrow wave function is a correlation factor used in quantum Monte Carlo methods to account for electron-electron interactions. 
# For a two-electron system like helium, the Jastrow factor is often expressed as an exponential function of the inter-electronic distance, 
# given by $\exp(\beta |r_1 - r_2|)$, where $\beta$ is a variational parameter. 
# The gradient and Laplacian of the Jastrow factor are important for calculating properties like the kinetic energy in quantum systems. 
# The gradient of the Jastrow factor with respect to the electron coordinates is given by the derivative of the exponential function, 
# and the Laplacian involves second derivatives. These derivatives are used to compute the kinetic energy and other observables in quantum systems.


class Jastrow:
    def __init__(self, beta=1):
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
        r_ee = self.get_r_ee(configs)[:, np.newaxis]
        grad = np.zeros_like(configs)
        grad[:, 0, :] = self.beta * r_vec / r_ee
        grad[:, 1, :] = -self.beta * r_vec / r_ee
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
        lap = np.zeros((configs.shape[0], configs.shape[1]))
        lap[:, 0] = self.beta * (ndim / r_ee - self.beta)
        lap[:, 1] = self.beta * (ndim / r_ee - self.beta)
        
        return lap


try:
    targets = process_hdf5_to_tuple('30.2', 3)
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