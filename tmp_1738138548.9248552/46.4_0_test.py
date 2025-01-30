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



def metropolis(configs, wf, tau, nsteps):
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    psi_cur = wf.value(poscur)

    for step in range(nsteps):
        # Propose a new position by applying a random logarithmic transformation
        log_shift = np.exp(np.sqrt(tau) * np.random.normal(size=poscur.shape))
        posnew = poscur * log_shift
        
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



# Background: 
# In quantum mechanics, the energy of a system can be decomposed into kinetic and potential components. 
# For a helium atom, the kinetic energy can be calculated using the Laplacian of the wave function, 
# while the potential energy consists of electron-ion and electron-electron interactions. 
# The Metropolis algorithm is a Monte Carlo method used to sample configurations of a system 
# according to a probability distribution, which in this case is related to the square of the wave function. 
# By running the Metropolis algorithm, we can generate a set of configurations that are used to estimate 
# the average kinetic and potential energies. The error bars are estimated using the standard error of the mean.


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
    
    # Initialize the wave function and Hamiltonian objects
    wf = Slater(alpha)
    hamiltonian = Hamiltonian(Z)
    
    # Perform Metropolis sampling to get new configurations
    sampled_configs = metropolis(configs, wf, tau, nsteps)
    
    # Calculate kinetic energy
    kinetic_energies = wf.kinetic(sampled_configs)
    
    # Calculate potential energies
    electron_ion_potentials = hamiltonian.potential_electron_ion(sampled_configs)
    electron_electron_potentials = hamiltonian.potential_electron_electron(sampled_configs)
    
    # Calculate mean energies
    kinetic_energy_mean = np.mean(kinetic_energies)
    electron_ion_potential_mean = np.mean(electron_ion_potentials)
    electron_electron_potential_mean = np.mean(electron_electron_potentials)
    
    # Calculate standard errors
    kinetic_energy_error = np.std(kinetic_energies) / np.sqrt(len(kinetic_energies))
    electron_ion_potential_error = np.std(electron_ion_potentials) / np.sqrt(len(electron_ion_potentials))
    electron_electron_potential_error = np.std(electron_electron_potentials) / np.sqrt(len(electron_electron_potentials))
    
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