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
        r = np.linalg.norm(configs, axis=2)
        val = np.exp(-self.alpha * np.sum(r, axis=1))
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)
        grad = -self.alpha * configs / r
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)
        ndim = configs.shape[2]
        lap = -self.alpha * (2 / r + self.alpha)  # Using the expression for the Laplacian of exp(-alpha * r) / r
        return np.sum(lap, axis=2)

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        laplacian_psi_over_psi = self.laplacian(configs)
        kin = -0.5 * np.sum(laplacian_psi_over_psi, axis=1)
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
        r = np.linalg.norm(configs, axis=2)  # calculate the norm of each electron position
        v_ei = -self.Z * np.sum(1 / r, axis=1)  # sum over electrons for each configuration
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        r12 = np.linalg.norm(configs[:, 0, :] - configs[:, 1, :], axis=1)  # distance between electron 1 and 2
        v_ee = 1 / r12  # potential energy between two electrons
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
        v = v_ei + v_ee  # total potential energy
        return v



def metropolis(configs, wf, hamiltonian, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        hamiltonian (Hamiltonian object): Hamiltonian class defined before
        tau (float): timestep for Metropolis moves
        nsteps (int): number of Metropolis steps
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    psi_cur = wf.value(poscur)

    for step in range(nsteps):
        # Propose a new move
        posnew = poscur + np.random.normal(scale=np.sqrt(tau), size=poscur.shape)
        psi_new = wf.value(posnew)

        # Compute Metropolis acceptance ratio
        acceptance_ratio = (psi_new / psi_cur) ** 2

        # Generate random numbers for acceptance
        random_numbers = np.random.rand(nconf)

        # Accept or reject the moves
        accept = acceptance_ratio > random_numbers
        poscur[accept] = posnew[accept]
        psi_cur[accept] = psi_new[accept]

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

    # Initialize the wave function and Hamiltonian
    wf = Slater(alpha)
    hamiltonian = Hamiltonian(Z)

    # Use the Metropolis algorithm to sample configurations
    sampled_configs = metropolis(configs, wf, hamiltonian, tau, nsteps)

    # Calculate kinetic energy
    kinetic_energies = wf.kinetic(sampled_configs)

    # Calculate electron-ion potential energy
    electron_ion_potentials = hamiltonian.potential_electron_ion(sampled_configs)

    # Calculate electron-electron potential energy
    electron_electron_potentials = hamiltonian.potential_electron_electron(sampled_configs)

    # Calculate means
    kinetic_energy_mean = np.mean(kinetic_energies)
    electron_ion_potential_mean = np.mean(electron_ion_potentials)
    electron_electron_potential_mean = np.mean(electron_electron_potentials)

    # Calculate standard errors
    kinetic_energy_error = np.std(kinetic_energies) / np.sqrt(len(kinetic_energies))
    electron_ion_potential_error = np.std(electron_ion_potentials) / np.sqrt(len(electron_ion_potentials))
    electron_electron_potential_error = np.std(electron_electron_potentials) / np.sqrt(len(electron_electron_potentials))

    # Store results in lists
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