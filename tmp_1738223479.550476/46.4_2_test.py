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

    # Calculate initial wave function values and their complex conjugates
    wf_values_curr = wf.value(poscur)
    wf_conj_curr = np.conjugate(wf_values_curr)

    for step in range(nsteps):
        # Propose new positions using normal distribution for each electron independently
        proposal = poscur + np.random.normal(scale=np.sqrt(tau), size=(nconf, nelec, ndim))
        
        # Compute wave function values for proposed configurations
        wf_values_prop = wf.value(proposal)
        wf_conj_prop = np.conjugate(wf_values_prop)

        # Calculate acceptance probability using squared magnitudes
        prob_ratio = np.abs(wf_values_prop * wf_conj_prop) / np.abs(wf_values_curr * wf_conj_curr)
        accept = prob_ratio > np.random.rand(nconf)

        # Update positions and wave function values where proposals are accepted
        poscur[accept] = proposal[accept]
        wf_values_curr[accept] = wf_values_prop[accept]
        wf_conj_curr[accept] = wf_conj_prop[accept]

    return poscur




def calc_energy(configs, nsteps, tau, alpha, Z):
    '''Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim) where nconf is the number of configurations, nelec is the number of electrons (2 for helium), ndim is the number of spatial dimensions (usually 3)
        nsteps (int): number of Metropolis steps
        tau (float): step size
        alpha (float): exponential decay factor for the wave function
        Z (int): atomic number
    Returns:
        energy (list of float): kinetic energy, electron-ion potential, and electron-electron potential
        error (list of float): error bars of kinetic energy, electron-ion potential, and electron-electron potential
    '''

    # Perform Metropolis sampling to update configurations
    updated_configs = metropolis(configs, alpha, tau, nsteps)

    # Calculate electron-ion and electron-electron distances using broadcasting
    electron_ion_distances = np.linalg.norm(updated_configs, axis=-1)
    electron_electron_distances = np.linalg.norm(updated_configs[:, 0, :] - updated_configs[:, 1, :], axis=-1)

    # Calculate wave function gradients for kinetic energy computation
    gradient_wf = -2 * alpha * updated_configs
    kinetic_energy = 0.5 * np.sum(gradient_wf**2, axis=(1, 2)) - alpha * np.sum(gradient_wf, axis=(1, 2))

    # Calculate potential energies
    electron_ion_potential = -Z * np.sum(1.0 / electron_ion_distances, axis=1)
    electron_electron_potential = 1.0 / electron_electron_distances

    # Compute means of each energy component
    kinetic_energy_mean = np.mean(kinetic_energy)
    electron_ion_potential_mean = np.mean(electron_ion_potential)
    electron_electron_potential_mean = np.mean(electron_electron_potential)

    # Calculate error bars using bootstrap resampling
    def bootstrap_error(data, n_samples=1000):
        resampled_means = [np.mean(np.random.choice(data, len(data))) for _ in range(n_samples)]
        return np.std(resampled_means)

    kinetic_energy_error = bootstrap_error(kinetic_energy)
    electron_ion_potential_error = bootstrap_error(electron_ion_potential)
    electron_electron_potential_error = bootstrap_error(electron_electron_potential)

    # Compile results
    energy = [kinetic_energy_mean, electron_ion_potential_mean, electron_electron_potential_mean]
    error = [kinetic_energy_error, electron_ion_potential_error, electron_electron_potential_error]

    return energy, error


try:
    targets = process_hdf5_to_tuple('46.4', 5)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    np.random.seed(0)
    assert cmp_tuple_or_list(calc_energy(np.random.randn(1000, 2, 3), nsteps=1000, tau=0.2, alpha=1, Z=2), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    np.random.seed(0)
    assert cmp_tuple_or_list(calc_energy(np.random.randn(1000, 2, 3), nsteps=1000, tau=0.2, alpha=2, Z=2), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    np.random.seed(0)
    assert cmp_tuple_or_list(calc_energy(np.random.randn(1000, 2, 3), nsteps=1000, tau=0.2, alpha=3, Z=2), target)

    target = targets[3]
    np.random.seed(0)
    energy, error = calc_energy(np.random.randn(1000, 2, 3), nsteps=10000, tau=0.05, alpha=1, Z=2)
    assert (energy[0]-error[0] < 1 and energy[0]+error[0] > 1, energy[1]-error[1] < -4 and energy[1]+error[1] > -4) == target

    target = targets[4]
    np.random.seed(0)
    energy, error = calc_energy(np.random.randn(1000, 2, 3), nsteps=10000, tau=0.05, alpha=2, Z=2)
    assert (energy[0]-error[0] < 4 and energy[0]+error[0] > 4, energy[1]-error[1] < -8 and energy[1]+error[1] > -8) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e