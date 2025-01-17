import numpy as np



# Background: The Slater wave function is a product of exponential functions representing the wave function of electrons in a helium atom. 
# It is given by the product of two exponential terms, each depending on the distance of an electron from the nucleus: 
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the two electrons from the nucleus.
# The gradient of the wave function with respect to the electron coordinates is used to calculate the force on the electrons.
# The Laplacian of the wave function is related to the kinetic energy of the electrons through the Schrödinger equation.
# The kinetic energy can be calculated using the formula: T = -0.5 * (laplacian psi) / psi.


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
        lap1 = self.alpha * (self.alpha * r1 - 2) / r1
        lap2 = self.alpha * (self.alpha * r2 - 2) / r2
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

targets = process_hdf5_to_tuple('46.1', 3)
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
