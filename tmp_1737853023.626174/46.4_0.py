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


# Background: The Metropolis algorithm is a Monte Carlo method used to sample from a probability distribution. 
# In the context of quantum mechanics, it is used to sample electron configurations according to the probability 
# distribution given by the square of the wave function, |psi|^2. The algorithm involves proposing a new configuration 
# by making a small random change to the current configuration and then deciding whether to accept or reject this 
# new configuration based on the ratio of the probabilities of the new and old configurations. The acceptance 
# probability is given by min(1, |psi_new|^2 / |psi_old|^2). If the new configuration is accepted, it becomes the 
# current configuration; otherwise, the old configuration is retained. This process is repeated for a specified 
# number of steps.


def metropolis(configs, wf, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        tau (float): timestep for proposal moves
        nsteps (int): number of Metropolis steps
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    if tau < 0:
        raise ValueError("Tau cannot be negative.")

    if not isinstance(configs, np.ndarray) or len(configs.shape) != 3:
        raise TypeError("configs must be a numpy array with three dimensions (nconf, nelec, ndim).")

    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    psi_cur = wf.value(poscur)
    
    for step in range(nsteps):
        # Propose a new position by adding a random displacement
        posnew = poscur + np.sqrt(tau) * np.random.normal(size=poscur.shape)
        
        # Calculate the new wave function value
        psi_new = wf.value(posnew)
        
        # Calculate the acceptance probability
        acceptance_prob = np.minimum(1, (psi_new / psi_cur) ** 2)
        
        # Generate random numbers for acceptance
        random_numbers = np.random.rand(nconf)
        
        # Accept or reject the new positions
        accept = random_numbers < acceptance_prob
        
        # Update positions and wave function values where accepted
        poscur[accept] = posnew[accept]
        psi_cur[accept] = psi_new[accept]
    
    return poscur



# Background: In quantum mechanics, the total energy of a system can be calculated as the sum of kinetic and potential energies. 
# For a helium atom, the kinetic energy is derived from the Laplacian of the wave function, as implemented in the Slater class. 
# The potential energy consists of electron-ion and electron-electron interactions, as implemented in the Hamiltonian class. 
# The Metropolis algorithm is used to sample electron configurations according to the probability distribution given by the 
# square of the wave function. By running the Metropolis algorithm, we can obtain a set of configurations that represent the 
# quantum state of the system. We then calculate the kinetic and potential energies for these configurations and estimate their 
# mean values and standard errors (error bars) to understand the energy distribution of the system.


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
    
    # Run the Metropolis algorithm to sample configurations
    sampled_configs = metropolis(configs, wf, tau, nsteps)
    
    # Calculate kinetic energy for sampled configurations
    kinetic_energies = wf.kinetic(sampled_configs)
    
    # Calculate potential energies for sampled configurations
    electron_ion_potentials = hamiltonian.potential_electron_ion(sampled_configs)
    electron_electron_potentials = hamiltonian.potential_electron_electron(sampled_configs)
    
    # Calculate mean energies
    kinetic_energy_mean = np.mean(kinetic_energies)
    electron_ion_potential_mean = np.mean(electron_ion_potentials)
    electron_electron_potential_mean = np.mean(electron_electron_potentials)
    
    # Calculate standard errors (error bars)
    kinetic_energy_error = np.std(kinetic_energies) / np.sqrt(len(kinetic_energies))
    electron_ion_potential_error = np.std(electron_ion_potentials) / np.sqrt(len(electron_ion_potentials))
    electron_electron_potential_error = np.std(electron_electron_potentials) / np.sqrt(len(electron_electron_potentials))
    
    # Compile results
    energy = [kinetic_energy_mean, electron_ion_potential_mean, electron_electron_potential_mean]
    error = [kinetic_energy_error, electron_ion_potential_error, electron_electron_potential_error]
    
    return energy, error

from scicode.parse.parse import process_hdf5_to_tuple
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
