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
# In the context of quantum Monte Carlo simulations, it is used to sample electron configurations according to the probability density given by the square of the wave function.
# The algorithm involves generating a trial move for the electron positions and accepting or rejecting this move based on the Metropolis acceptance criterion.
# This criterion uses the ratio of the probability density of the trial and current configurations, which is proportional to the square of the wave function values at these configurations.
# If the trial configuration is accepted, it becomes the new configuration; otherwise, the current configuration is retained.
# The timestep `tau` is used to scale the proposed moves, and the number of steps `nsteps` indicates how many Metropolis steps to perform.


def metropolis(configs, wf, tau, nsteps):
    '''Runs Metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        hamiltonian (Hamiltonian object): Hamiltonian class defined before
        tau (float): timestep for the Metropolis moves
        nsteps (int): number of Metropolis steps to perform
    Returns:
        poscur (np.array): final electron coordinates after Metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    wf_current = wf.value(poscur)
    
    for step in range(nsteps):
        # Generate trial moves
        trial_move = np.random.normal(scale=np.sqrt(tau), size=(nconf, nelec, ndim))
        postrial = poscur + trial_move

        # Calculate wave function value at trial positions
        wf_trial = wf.value(postrial)
        
        # Metropolis acceptance criterion
        acceptance_ratio = (wf_trial / wf_current) ** 2
        random_vals = np.random.rand(nconf)
        
        # Accept or reject moves
        accept = acceptance_ratio > random_vals
        poscur[accept] = postrial[accept]
        wf_current[accept] = wf_trial[accept]

    return poscur

from scicode.parse.parse import process_hdf5_to_tuple
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
