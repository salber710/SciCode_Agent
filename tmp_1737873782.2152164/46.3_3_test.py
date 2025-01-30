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
    poscur = configs.copy()  # Initial positions
    wf_current = wf.value(poscur)  # Evaluate the wavefunction at the current positions

    for step in range(nsteps):
        # Propose a move
        move = np.random.normal(scale=np.sqrt(tau), size=poscur.shape)  # Normally distributed random move
        posnew = poscur + move  # New positions

        # Evaluate wavefunction at the new positions
        wf_new = wf.value(posnew)

        # Calculate acceptance probability
        acceptance = np.square(wf_new / wf_current)

        # Compare with random numbers to decide whether to accept the move
        accept = acceptance >= np.random.rand(nconf)

        # Update positions and wavefunction values where the move was accepted
        poscur[accept] = posnew[accept]
        wf_current[accept] = wf_new[accept]

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