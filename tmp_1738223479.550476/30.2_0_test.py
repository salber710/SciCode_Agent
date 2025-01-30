from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


class Slater:
    def __init__(self, alpha):
        '''Args: 
            alpha: exponential decay factor
        '''
        self.alpha = alpha

    def _compute_distances(self, configs):
        return np.sqrt(np.sum(configs**2, axis=-1))

    def value(self, configs):
        '''Calculate unnormalized psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        distances = self._compute_distances(configs)
        total_distance = np.sum(distances, axis=1)
        val = np.exp(-self.alpha * total_distance)
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        distances = self._compute_distances(configs)
        grad = -self.alpha * configs / distances[:, :, np.newaxis]
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        distances = self._compute_distances(configs)
        lap = (self.alpha**2 - 2 * self.alpha / distances) / distances
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
# The Jastrow wave function introduces a correlation factor between electrons to account for electron-electron interactions 
# that the Slater determinant alone cannot capture. For two electrons, the Jastrow factor is an exponential function 
# of the distance between the two electrons, given by exp(beta * |r1 - r2|). 
# This Jastrow factor modifies the wave function by considering the relative separation of electrons, 
# adding flexibility to model electron correlation effects.
# The gradient and laplacian of the Jastrow factor are calculated with respect to electron coordinates. 
# The gradient of psi with respect to position is obtained by differentiating the Jastrow factor, 
# and the Laplacian involves the second derivative, which is related to the kinetic energy of the system.


class Jastrow:
    def __init__(self, beta=1):
        '''
        Initialize the Jastrow factor with a correlation parameter beta.
        Args:
            beta: exponential correlation factor
        '''
        self.beta = beta

    def get_r_vec(self, configs):
        '''Returns a vector pointing from r2 to r1, which is r_12 = [x1 - x2, y1 - y2, z1 - z2].
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): (nconf, ndim)
        '''
        return configs[:, 0, :] - configs[:, 1, :]

    def get_r_ee(self, configs):
        '''Returns the Euclidean distance from r2 to r1
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): (nconf,)
        '''
        r_vec = self.get_r_vec(configs)
        return np.linalg.norm(r_vec, axis=1)

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
        grad[:, 0, :] = self.beta * r_vec / r_ee  # Gradient with respect to r1
        grad[:, 1, :] = -self.beta * r_vec / r_ee  # Gradient with respect to r2
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
        lap = np.zeros((configs.shape[0], 2))
        
        # Laplacian with respect to r1
        lap[:, 0] = self.beta * (2 / r_ee - (self.beta) * (r_vec[:, 0]**2 + r_vec[:, 1]**2 + r_vec[:, 2]**2) / (r_ee**2))
        
        # Laplacian with respect to r2 (same as r1 due to symmetry)
        lap[:, 1] = lap[:, 0]

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