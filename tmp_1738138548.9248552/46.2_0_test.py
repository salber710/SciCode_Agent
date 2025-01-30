from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np


class Slater:
    def __init__(self, alpha):
        self.alpha = alpha

    def value(self, configs):
        r = np.sqrt(np.sum(configs**2, axis=2))
        psi = np.exp(-self.alpha * r).prod(axis=1)
        return psi

    def gradient(self, configs):
        r = np.sqrt(np.sum(configs**2, axis=2, keepdims=True))
        unit_vectors = configs / r
        grad_psi_psi = -self.alpha * unit_vectors
        return grad_psi_psi

    def laplacian(self, configs):
        r = np.sqrt(np.sum(configs**2, axis=2))
        lap_psi_psi = self.alpha**2 - 2 * self.alpha / r
        return lap_psi_psi

    def kinetic(self, configs):
        lap = self.laplacian(configs)
        kinetic_energy = -0.5 * np.sum(lap, axis=1)
        return kinetic_energy



# Background: In quantum mechanics, the Hamiltonian operator represents the total energy of a system. 
# For a helium atom, which consists of two electrons and a nucleus, the Hamiltonian includes kinetic energy 
# and potential energy terms. The potential energy is composed of electron-ion and electron-electron interactions. 
# The electron-ion potential is due to the attraction between each electron and the nucleus, which is modeled 
# by Coulomb's law as V_ei = -Z/r, where Z is the atomic number and r is the distance from the nucleus. 
# The electron-electron potential accounts for the repulsion between the two electrons, given by V_ee = 1/r_12, 
# where r_12 is the distance between the two electrons.


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
        # Calculate the distance of each electron from the nucleus (assumed at origin)
        r = np.sqrt(np.sum(configs**2, axis=2))
        # Electron-ion potential for each electron
        v_ei = -self.Z / r
        # Sum the potentials for both electrons
        v_ei_total = np.sum(v_ei, axis=1)
        return v_ei_total

    def potential_electron_electron(self, configs):
        '''Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Calculate the distance between the two electrons for each configuration
        r12 = np.sqrt(np.sum((configs[:, 0, :] - configs[:, 1, :])**2, axis=1))
        # Electron-electron potential
        v_ee = 1.0 / r12
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        # Calculate total potential energy as the sum of electron-ion and electron-electron potentials
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        v = v_ei + v_ee
        return v


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