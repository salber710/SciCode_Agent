import numpy as np



# Background: 
# A Slater wave function is a product of exponential functions for each electron, often used in quantum chemistry 
# models to describe the behavior of electrons in atoms or molecules. For helium, the wave function is 
# given by psi = exp(-αr1) * exp(-αr2), where α is an exponential decay factor, and r1, r2 are the distances of 
# the two electrons from the nucleus. 
# The gradient of the wave function divided by the wave function (grad(psi)/psi) is a vector whose components 
# are derived from the partial derivatives with respect to each spatial coordinate.
# The Laplacian of psi divided by psi (laplacian(psi)/psi) is a scalar that involves the second derivatives 
# of the wave function.
# The kinetic energy operator in quantum mechanics is related to the Laplacian via the formula: 
# T = -0.5 * (laplacian psi) / psi, which gives the kinetic energy per unit of the wave function psi.


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
        r = np.linalg.norm(configs, axis=-1)  # Calculate the distance of each electron from origin
        val = np.exp(-self.alpha * r[:, 0]) * np.exp(-self.alpha * r[:, 1])
        return val
    
    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.linalg.norm(configs, axis=-1, keepdims=True)
        grad = -self.alpha * configs / r
        return grad
    
    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.linalg.norm(configs, axis=-1)
        lap = -2 * self.alpha / r + self.alpha**2
        return lap
    
    def kinetic(self, configs):
        '''Calculate the kinetic energy / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        laplacian = self.laplacian(configs)
        kin = -0.5 * np.sum(laplacian, axis=-1)
        return kin

from scicode.parse.parse import process_hdf5_to_tuple
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
