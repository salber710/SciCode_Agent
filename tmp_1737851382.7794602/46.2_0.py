import numpy as np

# Background: 
# The Slater wave function for a helium atom is a product of exponential functions for each electron, 
# given by psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons 
# from the nucleus. The gradient of the wave function with respect to the electron coordinates is 
# calculated as the derivative of psi divided by psi, which simplifies to -alpha * r_hat, where r_hat 
# is the unit vector in the direction of the electron. The Laplacian of the wave function is the 
# divergence of the gradient, which for an exponential function results in a term involving the second 
# derivative. The kinetic energy operator in quantum mechanics is related to the Laplacian by 
# T = -0.5 * (hbar^2 / m) * Laplacian(psi) / psi, where hbar is the reduced Planck's constant and m is 
# the electron mass. In atomic units, hbar and m are set to 1, simplifying the kinetic energy to 
# -0.5 * Laplacian(psi) / psi.


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
        val = np.exp(-self.alpha * (r1 + r2))
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
        grad1 = -self.alpha * configs[:, 0, :] / np.where(r1 == 0, 1, r1)
        grad2 = -self.alpha * configs[:, 1, :] / np.where(r2 == 0, 1, r2)
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
        lap1 = self.alpha**2 - 2 * self.alpha / np.where(r1 == 0, np.inf, r1)
        lap2 = self.alpha**2 - 2 * self.alpha / np.where(r2 == 0, np.inf, r2)
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



# Background: In quantum mechanics, the potential energy of a system is crucial for determining the behavior of particles. 
# For a helium atom, which consists of two electrons and a nucleus, we consider two types of potential energies:
# 1. Electron-ion potential: This is the potential energy due to the attraction between each electron and the nucleus. 
#    It is given by V_ei = -Z/r, where Z is the atomic number (2 for helium) and r is the distance from the electron to the nucleus.
# 2. Electron-electron potential: This is the potential energy due to the repulsion between the two electrons. 
#    It is given by V_ee = 1/r_12, where r_12 is the distance between the two electrons.
# The total potential energy of the system is the sum of the electron-ion and electron-electron potentials.


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
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)
        v_ei = -self.Z * (1 / np.where(r1 == 0, np.inf, r1) + 1 / np.where(r2 == 0, np.inf, r2))
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        r12 = np.linalg.norm(configs[:, 0, :] - configs[:, 1, :], axis=1)
        v_ee = 1 / np.where(r12 == 0, np.inf, r12)
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        v = v_ei + v_ee
        return v

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('46.2', 3)
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
