import numpy as np

# Background: 
# The Slater wave function for a two-electron system like helium can be represented as the product of exponential functions:
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons from the nucleus.
# The gradient of psi with respect to the electron coordinates is given by the partial derivatives of psi.
# The Laplacian of psi involves taking the second partial derivatives of psi.
# For kinetic energy calculations in quantum mechanics, the kinetic energy operator is related to the Laplacian of the wave function.
# Specifically, the kinetic energy T can be calculated from the Laplacian as T = - (1/2) * (laplacian psi) / psi.


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
        r = np.linalg.norm(configs, axis=2)  # Calculate the distance from the nucleus for each electron
        val = np.exp(-self.alpha * np.sum(r, axis=1))  # Unnormalized wave function value
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.linalg.norm(configs, axis=2, keepdims=True)
        grad = -self.alpha * configs / r  # Gradient of psi with respect to each electron's position
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
        lap = -self.alpha**2 + self.alpha * (ndim - 1) / r  # Laplacian of psi
        return np.sum(lap, axis=2)  # Sum over dimensions for each electron

    def kinetic(self, configs):
        '''Calculate the kinetic energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)  # Kinetic energy from the Laplacian
        return kin


# Background: 
# In quantum mechanics, the Hamiltonian operator represents the total energy of a system, including both kinetic and potential energies.
# For a helium atom, the potential energy includes contributions from electron-ion and electron-electron interactions.
# The electron-ion potential energy is due to the attraction between the negatively charged electrons and the positively charged nucleus, given by V_ei = -Z / r_i, where Z is the atomic number and r_i is the distance of electron i from the nucleus.
# The electron-electron potential energy arises from the repulsion between the two electrons, given by V_ee = 1 / r_12, where r_12 is the distance between the two electrons.


class Hamiltonian:
    def __init__(self, Z):
        '''Z: atomic number
        '''
        self.Z = Z

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        r = np.linalg.norm(configs, axis=2)  # Calculate the distance from the nucleus for each electron
        v_ei = -self.Z * np.sum(1 / r, axis=1)  # Sum over electrons for each configuration
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        r12 = np.linalg.norm(configs[:, 0, :] - configs[:, 1, :], axis=1)  # Distance between two electrons
        v_ee = 1 / r12  # Electron-electron potential for each configuration
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
        v = v_ei + v_ee  # Total potential energy for each configuration
        return v


# Background: 
# The Metropolis algorithm is a Monte Carlo method used to sample from a probability distribution.
# In this context, it is used to sample electron configurations according to the probability density given by the square of the wave function, |psi|^2.
# The algorithm works by proposing a new configuration and deciding whether to accept or reject it based on a probability ratio.
# The proposal is usually done by adding a small random displacement to the current positions, scaled by a timestep parameter tau.
# The acceptance probability A is calculated as min(1, |psi_new/psi_old|^2).
# If a new configuration is accepted, it becomes the current configuration; otherwise, the old configuration is retained.


def metropolis(configs, wf, tau, nsteps):
    '''Runs Metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        hamiltonian (Hamiltonian object): Hamiltonian class defined before
        tau (float): timestep for proposal moves
        nsteps (int): number of Metropolis steps
    Returns:
        poscur (np.array): final electron coordinates after Metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    
    for step in range(nsteps):
        # Propose new moves
        proposal = poscur + np.sqrt(tau) * np.random.normal(size=(nconf, nelec, ndim))
        
        # Evaluate psi for current and proposed configurations
        psi_cur = wf.value(poscur)
        psi_prop = wf.value(proposal)
        
        # Calculate acceptance probabilities
        acceptance_ratio = (psi_prop / psi_cur) ** 2
        acceptance_prob = np.minimum(1, acceptance_ratio)
        
        # Decide whether to accept each proposed move
        accept = np.random.rand(nconf) < acceptance_prob
        
        # Update positions where moves are accepted
        poscur[accept] = proposal[accept]

    return poscur



# Background:
# In quantum mechanics simulations, estimating the properties of a system such as kinetic energy and potential energy
# requires accurate sampling of configurations. The Metropolis algorithm provides a mechanism to sample these configurations
# based on the probability distribution derived from the wave function. By using sampled configurations, we can calculate 
# the expectation values of kinetic and potential energies. The standard deviation of these expectation values across 
# configurations provides an estimate of the error (or error bars) for these quantities.


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
    
    # Calculate standard errors (as error bars)
    kinetic_energy_error = np.std(kinetic_energies) / np.sqrt(len(kinetic_energies))
    electron_ion_potential_error = np.std(electron_ion_potentials) / np.sqrt(len(electron_ion_potentials))
    electron_electron_potential_error = np.std(electron_electron_potentials) / np.sqrt(len(electron_electron_potentials))
    
    # Return energies and their errors
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
