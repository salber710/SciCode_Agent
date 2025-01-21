import numpy as np



# Background: 
# The Slater wave function is a product of exponential terms, one for each electron, representing a simple
# atomic orbital model. In this problem, we're dealing with a helium atom, which has two electrons. The wave 
# function for each electron is represented by exp(-alpha * r), where r is the distance of the electron from 
# the nucleus. The unnormalized wave function for the entire system is the product of these individual wave 
# functions.
#
# The gradient of a function is a vector of its partial derivatives. For the wave function psi, we compute:
# gradient_psi = -alpha * psi * (r_vector / r), where r_vector is the vector from the nucleus to the electron,
# and r is its magnitude.
# 
# The laplacian is the divergence of the gradient. For the exponential function, the laplacian of psi, divided 
# by psi, is given by: (laplacian_psi) / psi = alpha^2 - 2 * alpha / r. This result is due to the second 
# derivative terms simplifying given the form of the exponential function.
#
# The kinetic energy in quantum mechanics can be calculated using the formula:
# kinetic_energy = -0.5 * sum(laplacian_psi / psi) over all electrons.


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
        r = np.linalg.norm(configs, axis=2)  # Compute the distance of each electron from the nucleus
        val = np.exp(-self.alpha * np.sum(r, axis=1))  # Product of exponentials for both electrons
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)  # (nconf, nelec, 1)
        grad = -self.alpha * (configs / r)  # (nconf, nelec, ndim)
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.linalg.norm(configs, axis=2)  # (nconf, nelec)
        lap = self.alpha**2 - 2 * self.alpha / r  # (nconf, nelec)
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)  # (nconf, nelec)
        kin = -0.5 * np.sum(lap, axis=1)  # Sum over electrons for each configuration
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
