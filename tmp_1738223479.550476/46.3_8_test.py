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

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        # Vectorized approach to calculate electron distances from nucleus at origin
        distances_from_nucleus = np.sqrt(np.einsum('ijk,ijk->ij', configs, configs))
        # Calculate electron-ion potential using np.reciprocal for better precision
        v_ei = -self.Z * np.einsum('ij->i', np.reciprocal(distances_from_nucleus))
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Compute vector differences and squared distances between electrons
        diff_vectors = configs[:, 0, :] - configs[:, 1, :]
        squared_diffs = np.einsum('ij,ij->i', diff_vectors, diff_vectors)
        # Calculate electron-electron potential
        v_ee = np.reciprocal(np.sqrt(squared_diffs))
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)
        '''
        # Total potential energy as a sum of the two potential types
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        return np.add(v_ei, v_ee)




def metropolis(configs, wf, hamiltonian, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        hamiltonian (Hamiltonian object): Hamiltonian class defined before
        tau (float): timestep for proposal moves
        nsteps (int): number of Metropolis steps to perform
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)

    # Calculate initial wave function values
    wf_values_curr = wf.value(poscur)

    for step in range(nsteps):
        # Propose new positions using a mixture of uniform and exponential distributions
        uniform_displacement = (np.random.rand(nconf, nelec, ndim) - 0.5) * 2 * np.sqrt(tau)
        exponential_displacement = np.random.exponential(scale=np.sqrt(tau), size=(nconf, nelec, ndim))
        proposal = poscur + uniform_displacement * np.sign(np.random.randn(nconf, nelec, ndim)) + exponential_displacement

        # Calculate wave function values for the proposed positions
        wf_values_prop = wf.value(proposal)

        # Metropolis acceptance criterion
        prob_ratio = (np.abs(wf_values_prop) / np.abs(wf_values_curr)) ** 2
        accept = prob_ratio > np.random.rand(nconf)

        # Update positions and wave function values where proposals are accepted
        poscur[accept] = proposal[accept]
        wf_values_curr[accept] = wf_values_prop[accept]

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