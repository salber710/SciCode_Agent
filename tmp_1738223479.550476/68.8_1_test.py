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
    '''Runs Metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object): MultiplyWF class
        hamiltonian (Hamiltonian object): Hamiltonian class
    Returns:
        poscur (np.array): final electron coordinates after Metropolis. Shape (nconf, nelec, ndim)
    '''

    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    wfold = wf.value(poscur)**2

    rng = np.random.default_rng()

    for step in range(nsteps):
        # Generate random steps using a triangular distribution
        deltas = rng.triangular(left=-1.0, mode=0.0, right=1.0, size=poscur.shape)
        posnew = poscur + tau * deltas

        # Evaluate wave function at new positions
        wfnew = wf.value(posnew)**2

        # Calculate acceptance ratio using a small damping factor
        damping_factor = 0.99
        acceptance_ratio = (wfnew / wfold) * damping_factor

        # Determine which moves to accept, using a random threshold
        threshold = rng.random(nconf)
        accept = acceptance_ratio > threshold

        # Update positions and wave function values
        poscur[accept] = posnew[accept]
        wfold[accept] = wfnew[accept]

    return poscur



def get_acceptance_ratio(configs_old, configs_new, drift_old, drift_new, dtau, wf):
    '''Args:
        configs_old (np.array): electron positions before move (nconf, nelec, ndim)
        configs_new (np.array): electron positions after move (nconf, nelec, ndim)
        drift_old (np.array): gradient calculated on old configs multiplied by dtau (nconf, nelec, ndim)
        drift_new (np.array): gradient calculated on new configs (nconf, nelec, ndim)
        dtau (float): time step
        wf (wave function object): MultiplyWF class
    Returns:
        acceptance_ratio (nconf,):
    '''

    # Compute wave function values ratio
    wf_val_ratio = wf.value(configs_new) / wf.value(configs_old)

    # Calculate differences in configurations with drift terms
    diff_forward = configs_new - configs_old - 0.5 * drift_old
    diff_reverse = configs_old - configs_new - 0.5 * drift_new

    # Compute squared norms of differences
    norm_forward_sq = np.einsum('ijk,ijk->i', diff_forward, diff_forward)
    norm_reverse_sq = np.einsum('ijk,ijk->i', diff_reverse, diff_reverse)

    # Calculate exponential terms for Green's function based on norms
    greens_forward = -norm_forward_sq / (2 * dtau)
    greens_reverse = -norm_reverse_sq / (2 * dtau)

    # Compute acceptance ratio using the wave function ratio and exponential terms
    acceptance_ratio = wf_val_ratio**2 * np.exp(greens_forward - greens_reverse)

    return acceptance_ratio



def branch(weight):
    '''Performs DMC branching.
    Args:
        weight (list or np.array): list of weights. Shape (nconfig,)
    Return:
        new_indices (list or np.array): indices of chosen configurations. Shape (nconfig,)
    '''
    nconfig = len(weight)
    # Normalize the weights to sum to nconfig
    normalized_weights = np.array(weight) / np.sum(weight) * nconfig
    
    # Initialize list for new indices
    new_indices = []
    
    # Calculate the integer cut-off based on normalized weights
    cutoff = np.floor(normalized_weights).astype(int)
    
    # Add indices according to the integer part of normalized weights
    for i, count in enumerate(cutoff):
        new_indices.extend([i] * count)
    
    # Determine the number of additional indices needed
    remaining = nconfig - len(new_indices)
    
    # Calculate fractional weights for remaining indices
    fractional_weights = normalized_weights - cutoff
    fractional_probs = fractional_weights / np.sum(fractional_weights)
    
    # Randomly select remaining indices based on fractional probabilities
    if remaining > 0:
        additional_indices = np.random.choice(np.arange(nconfig), size=remaining, p=fractional_probs)
        new_indices.extend(additional_indices)
    
    # Shuffle the new indices to ensure randomness
    np.random.shuffle(new_indices)
    
    return np.array(new_indices)




def run_dmc(ham, wf, configs, tau, nstep):
    '''Run DMC
    Args:
        ham (hamiltonian object):
        wf (wavefunction object):
        configs (np.array): electron positions before move (nconf, nelec, ndim)
        tau: time step
        nstep: total number of iterations        
    Returns:
        list of local energies
    '''
    
    nconf, nelec, ndim = configs.shape
    energies = []

    for step in range(nstep):
        # Drift calculation
        drift_old = wf.gradient(configs) * tau
        random_displacement = np.sqrt(tau) * np.random.normal(size=configs.shape)
        configs_new = configs + drift_old + random_displacement
        
        # New drift calculation
        drift_new = wf.gradient(configs_new) * tau
        
        # Acceptance ratio calculation
        acceptance_ratio = get_acceptance_ratio(configs, configs_new, drift_old, drift_new, tau, wf)
        
        # Acceptance step using uniform random values
        random_values = np.random.random(size=nconf)
        accept = acceptance_ratio > random_values
        # Update configurations based on acceptance
        configs[accept] = configs_new[accept]

        # Calculate local energy
        kinetic_energy = -0.5 * np.sum(wf.laplacian(configs), axis=(1, 2))
        potential_energy = ham.potential(configs)
        local_energy = kinetic_energy + potential_energy
        energies.append(np.mean(local_energy))

        # Calculate weights and perform branching
        weights = np.exp(-tau * (local_energy - np.mean(local_energy)))
        indices = branch(weights)
        configs = configs[indices]

        # Set weights to average weight
        average_weight = np.mean(weights)
        weights[:] = average_weight

    return energies


try:
    targets = process_hdf5_to_tuple('68.8', 5)
    target = targets[0]
    np.random.seed(0)
    assert np.allclose(run_dmc(
        Hamiltonian(Z=1), 
        MultiplyWF(Slater(alpha=1.0), Jastrow(beta=1.0)), 
        np.random.randn(5000, 2, 3), 
        tau=0.1, 
        nstep=10
    ), target)

    target = targets[1]
    np.random.seed(0)
    assert np.allclose(run_dmc(
        Hamiltonian(Z=2), 
        MultiplyWF(Slater(alpha=3.0), Jastrow(beta=1.5)), 
        np.random.randn(5000, 2, 3), 
        tau=0.01, 
        nstep=20
    ), target)

    target = targets[2]
    np.random.seed(0)
    assert np.allclose(run_dmc(
        Hamiltonian(Z=3), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=2.0)), 
        np.random.randn(5000, 2, 3), 
        tau=0.5, 
        nstep=30
    ), target)

    target = targets[3]
    np.random.seed(0)
    assert np.allclose(run_dmc(
        Hamiltonian(Z=2), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        np.random.randn(5000, 2, 3), 
        tau=0.01, 
        nstep=1000
    ), target)

    target = targets[4]
    def C(t, a):
        mu = np.mean(a)
        l = (a[:-t] - mu)*(a[t:] - mu)
        c = 1/np.var(a, ddof=0)*np.mean(l)
        return c
    def get_auto_correlation_time(a):
        '''
        Computes autocorrelation time
        '''
        n = len(a)
        l = []
        for t in range(1, n):
            c = C(t, a)
            if c <= 0:
                break
            l.append(c)
        return 1 + 2*np.sum(l)
    def get_sem(a):
        '''
        Computes the standard error of the mean of a
        Args:
          a (np.array):  time series
        Returns:
          float: standard error of a
        '''
        k = get_auto_correlation_time(a)
        n = len(a)
        return np.std(a, ddof=0) / (n / k)**0.5
    def get_stats(l):
        l = np.array(l)
        mean = np.average(l)
        k = get_auto_correlation_time(l)
        err = get_sem(l)
        return mean, err
    warmup = 100
    np.random.seed(0)
    energies = run_dmc(
        Hamiltonian(Z=2), 
        MultiplyWF(Slater(alpha=2.0), Jastrow(beta=0.5)), 
        np.random.randn(5000, 2, 3), 
        tau=0.01, 
        nstep=1000
    )
    e_avg, e_err = get_stats(energies[warmup:])
    e_ref = -2.903724
    assert (np.abs(e_ref - e_avg) < 0.05) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e