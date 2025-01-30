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

    def _calc_distance_to_origin(self, electron_positions):
        '''Calculate distance of electrons from the origin (nucleus)'''
        return np.linalg.norm(electron_positions, axis=-1)

    def _calc_inter_electron_distance(self, electron1_pos, electron2_pos):
        '''Calculate distance between two electrons'''
        return np.linalg.norm(electron1_pos - electron2_pos, axis=-1)

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        # Calculate distances from electrons to the nucleus
        distances = self._calc_distance_to_origin(configs)
        # Sum potentials for each electron in each configuration
        electron_ion_potential = -self.Z * np.sum(1.0 / distances, axis=1)
        return electron_ion_potential

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Extract positions of each electron
        electron_positions_1 = configs[:, 0, :]
        electron_positions_2 = configs[:, 1, :]
        # Calculate the distance between the two electrons
        inter_electron_distance = self._calc_inter_electron_distance(electron_positions_1, electron_positions_2)
        # Calculate potential energy due to electron-electron repulsion
        v_ee = 1.0 / inter_electron_distance
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)
        '''
        # Compute total potential energy by summing individual potentials
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        total_potential_energy = v_ei + v_ee
        return total_potential_energy


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