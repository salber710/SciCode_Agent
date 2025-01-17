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


# Background: 
# The Jastrow wave function is a correlation factor used in quantum mechanics to account for electron-electron interactions.
# For a two-electron system, the Jastrow factor is given by psi = exp(beta * |r1 - r2|), where |r1 - r2| is the distance between the two electrons.
# The gradient of the Jastrow wave function with respect to the electron coordinates involves the derivative of the exponential function.
# The Laplacian of the Jastrow wave function involves second derivatives and is used to calculate the kinetic energy contribution from electron-electron interactions.


class Jastrow:
    def __init__(self, beta=1):
        '''
        Initialize the Jastrow wave function with a given beta parameter.
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
        r_ee = self.get_r_ee(configs)[:, np.newaxis]
        
        grad1 = self.beta * r_vec / r_ee
        grad2 = -grad1
        
        grad = np.stack((grad1, grad2), axis=1)
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
        
        # Calculate the Laplacian
        lap1 = self.beta * (2 / r_ee - (self.beta * r_vec**2).sum(axis=1) / r_ee**2)
        lap2 = lap1  # Symmetric for both electrons
        
        lap = np.stack((lap1, lap2), axis=1)
        return lap


# Background: In quantum mechanics, the multiplication of two wave functions is often used to describe systems where different factors contribute to the overall wave function. 
# When multiplying two wave functions, the resulting wave function's value is simply the product of the two individual wave functions' values. 
# The gradient of the product of two wave functions can be found using the product rule for differentiation: 
# (grad(psi1 * psi2)) / (psi1 * psi2) = (grad(psi1) / psi1) + (grad(psi2) / psi2).
# Similarly, the Laplacian of the product of two wave functions is given by:
# (laplacian(psi1 * psi2)) / (psi1 * psi2) = (laplacian(psi1) / psi1) + (laplacian(psi2) / psi2) + 2 * (grad(psi1) / psi1) * (grad(psi2) / psi2).
# The kinetic energy for the product of two wave functions can be derived from the Laplacian, as it is related to the second derivatives of the wave function.


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
        val = val1 * val2
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        grad = grad1 + grad2
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        lap1 = self.wf1.laplacian(configs)
        lap2 = self.wf2.laplacian(configs)
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        
        # Calculate the cross term: 2 * (grad1 * grad2)
        cross_term = 2 * np.sum(grad1 * grad2, axis=2)
        
        lap = lap1 + lap2 + cross_term
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin



# Background: In quantum mechanics, the Hamiltonian operator is used to describe the total energy of a system. 
# For a helium atom, the Hamiltonian includes both kinetic and potential energy terms. 
# The potential energy consists of two main components: the electron-ion potential and the electron-electron potential.
# The electron-ion potential is the Coulombic attraction between each electron and the nucleus, given by -Z/r_i, where Z is the atomic number and r_i is the distance of the electron from the nucleus.
# The electron-electron potential is the Coulombic repulsion between the two electrons, given by 1/|r1 - r2|, where |r1 - r2| is the distance between the two electrons.


class Hamiltonian:
    def __init__(self, Z):
        '''Z: atomic number'''
        self.Z = Z

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        # Calculate the distance of each electron from the nucleus
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)
        
        # Calculate the electron-ion potential for each configuration
        v_ei = -self.Z * (1/r1 + 1/r2)
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Calculate the distance between the two electrons
        r12 = np.linalg.norm(configs[:, 0, :] - configs[:, 1, :], axis=1)
        
        # Calculate the electron-electron potential for each configuration
        v_ee = 1/r12
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        # Calculate the total potential energy as the sum of electron-ion and electron-electron potentials
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        v = v_ei + v_ee
        return v


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('68.4', 3)
target = targets[0]

np.random.seed(0)
configs = np.random.normal(size=(1, 2, 3))
hamiltonian = Hamiltonian(Z=2)
assert np.allclose((hamiltonian.potential_electron_ion(configs), hamiltonian.potential_electron_electron(configs), 
                    hamiltonian.potential(configs)), target)
target = targets[1]

np.random.seed(0)
configs = np.random.normal(size=(2, 2, 3))
hamiltonian = Hamiltonian(Z=3)
assert np.allclose((hamiltonian.potential_electron_ion(configs), hamiltonian.potential_electron_electron(configs), 
                    hamiltonian.potential(configs)), target)
target = targets[2]

np.random.seed(0)
configs = np.random.normal(size=(3, 2, 3))
hamiltonian = Hamiltonian(Z=4)
assert np.allclose((hamiltonian.potential_electron_ion(configs), hamiltonian.potential_electron_electron(configs), 
                    hamiltonian.potential(configs)), target)
