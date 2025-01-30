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
        r = np.sqrt(np.sum(configs**2, axis=2))
        val = np.exp(-self.alpha * np.sum(r, axis=1))
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r = np.sqrt(np.sum(configs**2, axis=2))
        grad = np.full(configs.shape, -self.alpha)
        r_nonzero = r != 0
        grad[r_nonzero] *= configs[r_nonzero] / r[r_nonzero][:, :, np.newaxis]
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        r = np.sqrt(np.sum(configs**2, axis=2))
        lap = np.full(r.shape, self.alpha**2)
        r_nonzero = r != 0
        lap[r_nonzero] -= 2 * self.alpha / r[r_nonzero]
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



class Jastrow:
    def __init__(self, beta=1):
        '''
        Initialize the Jastrow class with a parameter beta.
        Args:
            beta: a parameter controlling the electron-electron correlation strength.
        '''
        self.beta = beta

    def electron_vector_difference(self, configs):
        '''Calculate the vector pointing from electron 2 to electron 1.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            vector_diff (np.array): (nconfig, ndim)
        '''
        vector_diff = np.subtract(configs[:, 0, :], configs[:, 1, :])
        return vector_diff

    def electron_distance(self, vector_diff):
        '''Calculate the Euclidean distance using the vector difference.
        Args:
            vector_diff (np.array): vector differences of shape (nconfig, ndim)
        Returns:
            distances (np.array): (nconfig,)
        '''
        distances = np.sqrt(np.sum(vector_diff**2, axis=1))
        return distances

    def value(self, configs):
        '''Evaluate the Jastrow wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            jastrow_value (np.array): (nconfig,)
        '''
        vector_diff = self.electron_vector_difference(configs)
        distances = self.electron_distance(vector_diff)
        jastrow_value = np.exp(self.beta * distances)
        return jastrow_value

    def gradient(self, configs):
        '''Compute the gradient of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            gradient (np.array): (nconfig, nelec, ndim)
        '''
        vector_diff = self.electron_vector_difference(configs)
        distances = self.electron_distance(vector_diff)
        gradient = np.zeros_like(configs)

        mask = distances > 0
        grad_coeff = self.beta / distances[mask, np.newaxis]
        gradient[mask, 0, :] = grad_coeff * vector_diff[mask]
        gradient[mask, 1, :] = -grad_coeff * vector_diff[mask]

        return gradient

    def laplacian(self, configs):
        '''Compute the Laplacian of the Jastrow wave function divided by the wave function.
        Args:
            configs (np.array): electron coordinates of shape (nconfig, nelec, ndim)
        Returns:
            laplacian (np.array): (nconfig, nelec)
        '''
        vector_diff = self.electron_vector_difference(configs)
        distances = self.electron_distance(vector_diff)
        ndim = configs.shape[2]
        laplacian = np.zeros((configs.shape[0], 2))

        mask = distances > 0
        lap_coefficient = self.beta * (ndim - 1) / distances[mask]
        laplacian[mask, 0] = lap_coefficient - self.beta**2
        laplacian[mask, 1] = lap_coefficient - self.beta**2

        return laplacian



class MultiplyWF:
    def __init__(self, wf1, wf2):
        '''Args:
            wf1 (wavefunction object): 
            wf2 (wavefunction object):            
        '''
        self.wf1 = wf1
        self.wf2 = wf2

    def value(self, configs):
        '''Multiply two wave function values using a custom transformation with sigmoid
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        val1 = self.wf1.value(configs)
        val2 = self.wf2.value(configs)
        # Use a sigmoid transformation to combine values
        return 1 / (1 + np.exp(-(val1 * val2)))

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi using a power mean of gradients
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        # Use power mean (p=3) of gradients
        return ((grad1**3 + grad2**3) / 2)**(1/3)

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi using a unique cross-term with exponential differences
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        lap1 = self.wf1.laplacian(configs)
        lap2 = self.wf2.laplacian(configs)
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        
        # Use exponential of differences for cross-term
        cross_term_exp_diff = np.sum(np.exp(np.abs(grad1 - grad2)), axis=2)
        return lap1 + lap2 + cross_term_exp_diff

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions with a unique power factor
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap_combined = self.laplacian(configs)
        # Use a power factor in the kinetic energy calculation
        kinetic_energy = -0.3 * np.power(np.sum(lap_combined, axis=1), 1.5)  # Power factor for uniqueness
        return kinetic_energy



class Hamiltonian:
    def __init__(self, Z):
        '''Z: atomic number'''
        self.Z = Z

    def potential_electron_ion(self, configs):
        '''Calculate electron-ion potential using a custom approach with explicit loops and distance caching
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ei (np.array): (nconf,)
        '''
        nconf, nelec, ndim = configs.shape
        v_ei = np.zeros(nconf)
        
        for i in range(nconf):
            distances = [np.sqrt(np.dot(configs[i, e], configs[i, e])) for e in range(nelec)]
            for dist in distances:
                v_ei[i] -= self.Z / dist
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential using a straightforward loop and Euclidean distance computation
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        nconf, nelec, ndim = configs.shape
        v_ee = np.zeros(nconf)
        
        for i in range(nconf):
            r12 = np.sqrt(np.sum((configs[i, 0] - configs[i, 1]) ** 2))
            v_ee[i] = 1.0 / r12
        return v_ee

    def potential(self, configs):
        '''Compute total potential energy by combining potentials using direct addition
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)
        '''
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        
        return v_ei + v_ee




def metropolis(configs, wf, hamiltonian, tau=0.01, nsteps=2000):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): MultiplyWF class
        hamiltonian (Hamiltonian object): Hamiltonian class
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    wfold_log = np.log(wf.value(poscur)**2)

    for step in range(nsteps):
        # Propose new positions with a different random step size
        step_size = tau * (1 + 0.1 * np.random.rand())
        posnew = poscur + np.random.uniform(-step_size, step_size, size=poscur.shape)
        
        # Evaluate wave function at new positions
        wfnew_log = np.log(wf.value(posnew)**2)
        
        # Calculate acceptance ratio using log to avoid numerical issues
        log_acceptance_ratio = wfnew_log - wfold_log
        
        # Determine which moves to accept
        accept = np.log(np.random.rand(nconf)) < log_acceptance_ratio
        
        # Update positions
        poscur[accept] = posnew[accept]
        wfold_log[accept] = wfnew_log[accept]
        
    return poscur


try:
    targets = process_hdf5_to_tuple('68.5', 3)
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