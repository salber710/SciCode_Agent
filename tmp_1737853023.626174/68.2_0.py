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
        nonzero_mask = r_ee != 0
        grad[nonzero_mask, 0, :] = self.beta * r_vec[nonzero_mask] / r_ee[nonzero_mask]
        grad[nonzero_mask, 1, :] = -self.beta * r_vec[nonzero_mask] / r_ee[nonzero_mask]
        
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs).reshape(-1, 1)
        ndim = configs.shape[2]
        
        lap = np.zeros((configs.shape[0], configs.shape[1]))
        
        # Avoid division by zero
        nonzero_mask = r_ee != 0
        lap[nonzero_mask, 0] = self.beta * (ndim - 1) / r_ee[nonzero_mask] - self.beta**2
        lap[nonzero_mask, 1] = self.beta * (ndim - 1) / r_ee[nonzero_mask] - self.beta**2
        
        return lap

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('68.2', 3)
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
