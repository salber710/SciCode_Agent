import numpy as np



# Background: 
# The Slater wave function for a two-electron system like helium is given by the product of exponential functions: 
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons from the nucleus.
# The gradient of the wave function with respect to the electron coordinates is given by the partial derivatives of psi.
# The gradient of psi divided by psi is a vector field that points in the direction of the greatest rate of increase of psi.
# The Laplacian of psi divided by psi involves the second derivatives and is related to the curvature of the wave function.
# The kinetic energy operator in quantum mechanics is related to the Laplacian and is given by -0.5 * (laplacian psi) / psi.


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
        val = np.exp(-self.alpha * r1) * np.exp(-self.alpha * r2)
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
        
        grad1 = -self.alpha * configs[:, 0, :] / r1
        grad2 = -self.alpha * configs[:, 1, :] / r2
        
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
        
        lap1 = self.alpha**2 - 2 * self.alpha / r1
        lap2 = self.alpha**2 - 2 * self.alpha / r2
        
        lap = np.stack((lap1, lap2), axis=1)
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


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('68.1', 3)
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
