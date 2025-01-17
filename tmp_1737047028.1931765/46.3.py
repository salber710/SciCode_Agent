import numpy as np

# Background: The Slater wave function is a product of exponential functions representing the wave function of electrons in a helium atom. 
# It is given by the product of two exponential terms, each depending on the distance of an electron from the nucleus: 
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the two electrons from the nucleus.
# The gradient of the wave function with respect to the electron coordinates is used to calculate the force on the electrons.
# The Laplacian of the wave function is related to the kinetic energy of the electrons through the Schrödinger equation.
# The kinetic energy can be calculated using the formula: T = -0.5 * (laplacian psi) / psi.


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
        val = np.exp(-self.alpha * r1) * np.exp(-self.alpha * r2)
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
        grad1 = -self.alpha * configs[:, 0, :] / r1
        grad2 = -self.alpha * configs[:, 1, :] / r2
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
        lap1 = self.alpha * (self.alpha * r1 - 2) / r1
        lap2 = self.alpha * (self.alpha * r2 - 2) / r2
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


# Background: In quantum mechanics, the Hamiltonian operator represents the total energy of a system, including both kinetic and potential energies. 
# For a helium atom, the potential energy consists of two main components: the electron-ion potential and the electron-electron potential.
# The electron-ion potential is the Coulombic attraction between each electron and the nucleus, which is given by V_ei = -Z/r, where Z is the atomic number and r is the distance from the nucleus.
# The electron-electron potential is the Coulombic repulsion between the two electrons, given by V_ee = 1/r_12, where r_12 is the distance between the two electrons.
# The total potential energy is the sum of the electron-ion and electron-electron potentials.


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
        v_ei = -self.Z * (1/r1 + 1/r2)
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        r12 = np.linalg.norm(configs[:, 0, :] - configs[:, 1, :], axis=1)
        v_ee = 1/r12
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
# distribution given by the square of the wave function. The algorithm involves proposing a new configuration 
# by making a small random change to the current configuration, and then accepting or rejecting this new configuration 
# based on the ratio of the probabilities of the new and old configurations. The acceptance criterion is typically 
# based on the Metropolis acceptance rule, which uses the exponential of the negative change in energy, scaled by 
# a parameter called the timestep (tau). This method allows for efficient sampling of configurations that contribute 
# significantly to the wave function.


def metropolis(configs, wf, tau, nsteps):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): Slater class defined before
        tau (float): timestep for the Metropolis algorithm
        nsteps (int): number of Metropolis steps to perform
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    psi_cur = wf.value(poscur)
    
    for step in range(nsteps):
        # Propose a new position by adding a small random displacement
        posnew = poscur + np.sqrt(tau) * np.random.randn(nconf, nelec, ndim)
        
        # Calculate the new wave function value
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
