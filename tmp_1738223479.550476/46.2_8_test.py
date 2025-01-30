from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


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
        # Compute distances using np.apply_along_axis for a different approach
        distances = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 2, configs)
        # Calculate the wave function value
        val = np.exp(-self.alpha * np.sum(distances, axis=1))
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        # Compute distances using np.apply_along_axis
        distances = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 2, configs)[:, :, np.newaxis]
        # Calculate the gradient
        grad = -self.alpha * configs / distances
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        # Compute distances using np.apply_along_axis
        distances = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 2, configs)
        # Calculate the laplacian
        lap = self.alpha**2 - 2 * self.alpha / distances
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        # Calculate kinetic energy from the laplacian
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin




class Hamiltonian:
    def __init__(self, Z):
        '''Z: atomic number'''
        self.Z = Z

    def _distance_from_origin(self, positions):
        '''Calculate Euclidean distance from the origin (nucleus)'''
        return np.sqrt(np.einsum('ijk,ijk->ij', positions, positions))

    def _distance_between_electrons(self, pos1, pos2):
        '''Calculate Euclidean distance between two electron positions'''
        difference = pos1 - pos2
        return np.sqrt(np.einsum('ij,ij->i', difference, difference))

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        # Calculate distance from each electron to the nucleus
        distances = self._distance_from_origin(configs)
        # Calculate and sum electron-ion potentials
        v_ei = -self.Z * np.sum(np.divide(1.0, distances), axis=1)
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Calculate distance between the two electrons
        electron1_positions = configs[:, 0, :]
        electron2_positions = configs[:, 1, :]
        r12 = self._distance_between_electrons(electron1_positions, electron2_positions)
        # Calculate electron-electron potential
        v_ee = np.reciprocal(r12)
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        # Sum electron-ion and electron-electron potentials
        v_total = np.add(self.potential_electron_ion(configs), self.potential_electron_electron(configs))
        return v_total


try:
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e