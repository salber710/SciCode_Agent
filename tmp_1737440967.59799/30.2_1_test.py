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



# Background: The Jastrow wave function is used in quantum chemistry and physics to describe electron correlations in a system. 
# It is a correlation factor that accounts for electron-electron interactions, which are not captured by simple Slater determinants. 
# For helium, the Jastrow wave function is given by psi = exp(β |r1 - r2|), where β is a variational parameter, and r1, r2 are 
# the positions of the two electrons. The gradient of the wave function divided by the wave function, (grad(psi)/psi), 
# is a vector that involves the derivatives with respect to each electron's coordinates. 
# The Laplacian of psi divided by psi, (laplacian(psi)/psi), involves the second derivatives and is used for calculating 
# the kinetic energy contribution due to electron-electron interactions.


class Jastrow:
    def __init__(self, beta=1):
        '''
        Args:
            beta: Variational parameter for the Jastrow factor
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
        r_ee = np.linalg.norm(r_vec, axis=-1)
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

        grad_psi_psi = np.zeros_like(configs)
        grad_psi_psi[:, 0, :] = self.beta * r_vec / r_ee[:, np.newaxis]
        grad_psi_psi[:, 1, :] = -self.beta * r_vec / r_ee[:, np.newaxis]

        return grad_psi_psi

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)
        '''
        r_ee = self.get_r_ee(configs)
        r_vec = self.get_r_vec(configs)

        lap = np.zeros((configs.shape[0], 2))
        
        # Laplacian for electron 1
        lap[:, 0] = self.beta * (2.0 / r_ee - (self.beta * r_vec**2).sum(axis=-1) / r_ee**2)

        # Laplacian for electron 2
        lap[:, 1] = self.beta * (2.0 / r_ee - (self.beta * r_vec**2).sum(axis=-1) / r_ee**2)
        
        return lap

from scicode.parse.parse import process_hdf5_to_tuple
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
