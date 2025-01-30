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



class Hamiltonian:
    def __init__(self, Z):
        self.Z = Z

    def potential_electron_ion(self, configs):
        # Calculate the electron-ion potential using a direct computation of the inverse of the distances
        distances = np.linalg.norm(configs, axis=2)
        v_ei = -self.Z / np.maximum(distances, 1e-12)  # Avoid division by zero by setting a minimum distance
        return np.sum(v_ei, axis=1)

    def potential_electron_electron(self, configs):
        # Calculate the electron-electron potential using a direct computation of the inverse of the distance
        delta_r = configs[:, 0, :] - configs[:, 1, :]
        r_ee = np.linalg.norm(delta_r, axis=1)
        v_ee = 1 / np.maximum(r_ee, 1e-12)  # Avoid division by zero by setting a minimum distance
        return v_ee

    def potential(self, configs):
        # Calculate the total potential energy for each configuration
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        total_potential = v_ei + v_ee
        return total_potential



# Background: The Metropolis algorithm is a Monte Carlo method used to sample from a probability distribution. 
# In the context of quantum mechanics, it is used to sample electron configurations according to the probability 
# distribution given by the square of the wave function. The algorithm involves proposing a new configuration 
# by making a small random change to the current configuration, and then accepting or rejecting this new 
# configuration based on the Metropolis acceptance criterion. The acceptance criterion is based on the ratio 
# of the probabilities of the new and old configurations, which in this case is the ratio of the squares of 
# the wave function values at these configurations. The parameter `tau` is related to the step size of the 
# random changes, and `nsteps` is the number of Metropolis steps to perform.


def metropolis(configs, wf, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        hamiltonian (Hamiltonian object): Hamiltonian class defined before
        tau (float): timestep for the Metropolis algorithm
        nsteps (int): number of Metropolis steps to perform
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    psi_cur = wf.value(poscur)

    for step in range(nsteps):
        # Propose a new position by adding a random displacement
        posnew = poscur + np.sqrt(tau) * np.random.normal(size=poscur.shape)
        
        # Calculate the wave function value at the new position
        psi_new = wf.value(posnew)
        
        # Calculate the acceptance probability
        acceptance_prob = (psi_new / psi_cur) ** 2
        
        # Generate random numbers for acceptance
        random_numbers = np.random.rand(nconf)
        
        # Accept or reject the new positions
        accept = acceptance_prob > random_numbers
        poscur[accept] = posnew[accept]
        psi_cur[accept] = psi_new[accept]

    return poscur


try:
    targets = process_hdf5_to_tuple('46.3', 3)
    target = targets[0]
    wf = Slater(alpha=1)
    np.random.seed(0)
    assert np.allclose(metropolis(np.random.normal(size=(1, 2, 3)), wf, tau=0.01, nsteps=2000), target)

    target = targets[1]
    wf = Slater(alpha=1)
    np.random.seed(0)
    assert np.allclose(metropolis(np.random.normal(size=(2, 2, 3)), wf, tau=0.01, nsteps=2000), target)

    target = targets[2]
    wf = Slater(alpha=1)
    np.random.seed(0)
    assert np.allclose(metropolis(np.random.normal(size=(3, 2, 3)), wf, tau=0.01, nsteps=2000), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e