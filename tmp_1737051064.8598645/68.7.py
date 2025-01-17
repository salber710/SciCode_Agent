import numpy as np

# Background: 
# The Slater wave function for a two-electron system like helium is given by the product of exponential functions: 
# psi = exp(-alpha * r1) * exp(-alpha * r2), where r1 and r2 are the distances of the electrons from the nucleus.
# The gradient of the wave function with respect to the electron coordinates is given by the partial derivatives of psi.
# The gradient of psi divided by psi is a vector field that points in the direction of the greatest rate of increase of psi.
# The Laplacian of psi divided by psi involves the second derivatives and is related to the curvature of the wave function.
# The kinetic energy operator in quantum mechanics is related to the Laplacian and is given by -0.5 * (laplacian psi) / psi.


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
        
        lap1 = self.alpha**2 - 2 * self.alpha / r1
        lap2 = self.alpha**2 - 2 * self.alpha / r2
        
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


# Background: 
# The Jastrow wave function is a correlation factor used in quantum mechanics to account for electron-electron interactions.
# For a two-electron system, the Jastrow factor is given by psi = exp(beta * |r1 - r2|), where |r1 - r2| is the distance between the two electrons.
# The gradient of the Jastrow wave function with respect to the electron coordinates involves the derivative of the exponential function.
# The Laplacian of the Jastrow wave function involves second derivatives and is used to calculate the kinetic energy contribution from electron-electron interactions.


class Jastrow:
    def __init__(self, beta=1):
        '''
        Initialize the Jastrow wave function with a given beta parameter.
        Args:
            beta: correlation factor
        '''
        self.beta = beta

    def get_r_vec(self, configs):
        '''Returns a vector pointing from r2 to r1, which is r_12 = [x1 - x2, y1 - y2, z1 - z2].
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_vec (np.array): (nconf, ndim)
        '''
        r_vec = configs[:, 0, :] - configs[:, 1, :]
        return r_vec

    def get_r_ee(self, configs):
        '''Returns the Euclidean distance from r2 to r1
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            r_ee (np.array): (nconf,)
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = np.linalg.norm(r_vec, axis=1)
        return r_ee

    def value(self, configs):
        '''Calculate Jastrow factor
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns 
            jast (np.array): (nconf,)
        '''
        r_ee = self.get_r_ee(configs)
        jast = np.exp(self.beta * r_ee)
        return jast

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)[:, np.newaxis]
        
        grad1 = self.beta * r_vec / r_ee
        grad2 = -grad1
        
        grad = np.stack((grad1, grad2), axis=1)
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array):  (nconf, nelec)        
        '''
        r_vec = self.get_r_vec(configs)
        r_ee = self.get_r_ee(configs)
        
        # Calculate the Laplacian
        lap1 = self.beta * (2 / r_ee - (self.beta * r_vec**2).sum(axis=1) / r_ee**2)
        lap2 = lap1  # Symmetric for both electrons
        
        lap = np.stack((lap1, lap2), axis=1)
        return lap


# Background: In quantum mechanics, the multiplication of two wave functions is often used to describe systems where different factors contribute to the overall wave function. 
# When multiplying two wave functions, the resulting wave function's value is simply the product of the two individual wave functions' values. 
# The gradient of the product of two wave functions can be found using the product rule for differentiation: 
# (grad(psi1 * psi2)) / (psi1 * psi2) = (grad(psi1) / psi1) + (grad(psi2) / psi2).
# Similarly, the Laplacian of the product of two wave functions is given by:
# (laplacian(psi1 * psi2)) / (psi1 * psi2) = (laplacian(psi1) / psi1) + (laplacian(psi2) / psi2) + 2 * (grad(psi1) / psi1) * (grad(psi2) / psi2).
# The kinetic energy for the product of two wave functions can be derived from the Laplacian, as it is related to the second derivatives of the wave function.


class MultiplyWF:
    def __init__(self, wf1, wf2):
        '''Args:
            wf1 (wavefunction object): 
            wf2 (wavefunction object):            
        '''
        self.wf1 = wf1
        self.wf2 = wf2

    def value(self, configs):
        '''Multiply two wave function values
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            val (np.array): (nconf,)
        '''
        val1 = self.wf1.value(configs)
        val2 = self.wf2.value(configs)
        val = val1 * val2
        return val

    def gradient(self, configs):
        '''Calculate (gradient psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            grad (np.array): (nconf, nelec, ndim)
        '''
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        grad = grad1 + grad2
        return grad

    def laplacian(self, configs):
        '''Calculate (laplacian psi) / psi of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            lap (np.array): (nconf, nelec)
        '''
        lap1 = self.wf1.laplacian(configs)
        lap2 = self.wf2.laplacian(configs)
        grad1 = self.wf1.gradient(configs)
        grad2 = self.wf2.gradient(configs)
        
        # Calculate the cross term: 2 * (grad1 * grad2)
        cross_term = 2 * np.sum(grad1 * grad2, axis=2)
        
        lap = lap1 + lap2 + cross_term
        return lap

    def kinetic(self, configs):
        '''Calculate the kinetic energy of the multiplication of two wave functions
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            kin (np.array): (nconf,)
        '''
        lap = self.laplacian(configs)
        kin = -0.5 * np.sum(lap, axis=1)
        return kin


# Background: In quantum mechanics, the Hamiltonian operator is used to describe the total energy of a system. 
# For a helium atom, the Hamiltonian includes both kinetic and potential energy terms. 
# The potential energy consists of two main components: the electron-ion potential and the electron-electron potential.
# The electron-ion potential is the Coulombic attraction between each electron and the nucleus, given by -Z/r_i, where Z is the atomic number and r_i is the distance of the electron from the nucleus.
# The electron-electron potential is the Coulombic repulsion between the two electrons, given by 1/|r1 - r2|, where |r1 - r2| is the distance between the two electrons.


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
        # Calculate the distance of each electron from the nucleus
        r1 = np.linalg.norm(configs[:, 0, :], axis=1)
        r2 = np.linalg.norm(configs[:, 1, :], axis=1)
        
        # Calculate the electron-ion potential for each configuration
        v_ei = -self.Z * (1/r1 + 1/r2)
        return v_ei

    def potential_electron_electron(self, configs):
        '''Calculate electron-electron potential
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v_ee (np.array): (nconf,)
        '''
        # Calculate the distance between the two electrons
        r12 = np.linalg.norm(configs[:, 0, :] - configs[:, 1, :], axis=1)
        
        # Calculate the electron-electron potential for each configuration
        v_ee = 1/r12
        return v_ee

    def potential(self, configs):
        '''Total potential energy
        Args:
            configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        Returns:
            v (np.array): (nconf,)        
        '''
        # Calculate the total potential energy as the sum of electron-ion and electron-electron potentials
        v_ei = self.potential_electron_ion(configs)
        v_ee = self.potential_electron_electron(configs)
        v = v_ei + v_ee
        return v


# Background: 
# The Metropolis algorithm is a Monte Carlo method used to sample from a probability distribution. 
# In the context of quantum mechanics, it is used to sample electron configurations according to the probability density given by the square of the wave function.
# The algorithm involves proposing a new configuration by making a small random move from the current configuration.
# The acceptance of the new configuration is determined by the Metropolis criterion, which involves the ratio of the wave function values at the new and old configurations.
# If the new configuration is accepted, it becomes the current configuration; otherwise, the old configuration is retained.
# This process is repeated for a specified number of steps to generate a sequence of configurations that sample the desired distribution.


def metropolis(configs, wf, tau=0.01, nsteps=2000):
    '''Runs metropolis sampling
    Args:
        configs (np.array): electron coordinates of shape (nconf, nelec, ndim)
        wf (wavefunction object):  MultiplyWF class      
    Returns:
        poscur (np.array): final electron coordinates after metropolis. Shape (nconf, nelec, ndim)
    '''
    nconf, nelec, ndim = configs.shape
    poscur = np.copy(configs)
    wfcurr = wf.value(poscur)
    
    for step in range(nsteps):
        # Propose a new position by adding a small random displacement
        posnew = poscur + np.sqrt(tau) * np.random.normal(size=poscur.shape)
        
        # Calculate the wave function value at the new position
        wfnew = wf.value(posnew)
        
        # Calculate the Metropolis acceptance ratio
        ratio = (wfnew / wfcurr)**2
        
        # Generate random numbers for acceptance criterion
        accept = np.random.rand(nconf)
        
        # Accept or reject the new positions
        mask = accept < ratio
        poscur[mask] = posnew[mask]
        wfcurr[mask] = wfnew[mask]
    
    return poscur


# Background: 
# In the Diffusion Monte Carlo (DMC) algorithm, the acceptance ratio is used to determine whether a proposed move in the electron configuration space should be accepted or rejected.
# The acceptance ratio is calculated using the Metropolis-Hastings criterion, which involves the ratio of the probability densities of the new and old configurations.
# In DMC, the drift term is used to guide the random walk towards regions of higher probability density, improving the efficiency of the sampling.
# The acceptance ratio for the drift part includes contributions from both the wave function values and the drift vectors, which are related to the gradient of the wave function.
# The acceptance ratio is given by:
# A = min(1, (psi_new / psi_old)^2 * exp(-0.5 * (drift_new^2 - drift_old^2) / dtau + (configs_new - configs_old) * (drift_new + drift_old) / dtau))


def get_acceptance_ratio(configs_old, configs_new, drift_old, drift_new, dtau, wf):
    '''Args:
        configs_old (np.array): electron positions before move (nconf, nelec, ndim)
        configs_new (np.array): electron positions after  move (nconf, nelec, ndim)
        drift_old (np.array): gradient calculated on old configs multiplied by dtau (nconf, nelec, ndim)
        drift_new (np.array): gradient calculated on new configs (nconf, nelec, ndim)
        dtau (float): time step
        wf (wave function object): MultiplyWF class
    Returns:
        acceptance_ratio (nconf,):
    '''
    # Calculate the wave function values at the old and new configurations
    psi_old = wf.value(configs_old)
    psi_new = wf.value(configs_new)
    
    # Calculate the ratio of the wave function values squared
    psi_ratio = (psi_new / psi_old)**2
    
    # Calculate the drift terms
    drift_diff = (configs_new - configs_old) * (drift_new + drift_old) / dtau
    drift_square_diff = 0.5 * (np.sum(drift_new**2, axis=(1, 2)) - np.sum(drift_old**2, axis=(1, 2))) / dtau
    
    # Calculate the acceptance ratio
    exponent = -drift_square_diff + np.sum(drift_diff, axis=(1, 2))
    acceptance_ratio = psi_ratio * np.exp(exponent)
    
    # Ensure the acceptance ratio is at most 1
    acceptance_ratio = np.minimum(1, acceptance_ratio)
    
    return acceptance_ratio



# Background: 
# In Diffusion Monte Carlo (DMC), branching is a process used to manage the population of configurations (or walkers) based on their weights.
# Each configuration has an associated weight that reflects its importance or contribution to the overall simulation.
# The branching process involves selecting configurations to keep or replicate based on their weights, ensuring that the total number of configurations remains constant.
# This is typically done by interpreting the weights as probabilities and using them to perform a stochastic selection of configurations.
# The goal is to preferentially keep configurations with higher weights, which are more likely to contribute to the correct ground state energy.
# A common method for branching is to use a stochastic universal sampling technique, which ensures that the expected number of copies of each configuration is proportional to its weight.


def branch(weight):
    '''Performs DMC branching.
    Args:
        weight (list or np.array): list of weights. Shape (nconfig,)
    Return:
        new_indices (list or np.array): indices of chosen configurations. Shape (nconfig,)
    '''
    nconfig = len(weight)
    cumulative_weight = np.cumsum(weight)
    total_weight = cumulative_weight[-1]
    
    # Generate a random starting point
    start = np.random.uniform(0, total_weight / nconfig)
    
    # Create an array of equally spaced points
    points = start + np.arange(nconfig) * (total_weight / nconfig)
    
    # Determine which configurations to keep
    new_indices = np.searchsorted(cumulative_weight, points)
    
    return new_indices


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('68.7', 3)
target = targets[0]

np.random.seed(0)
assert np.allclose(branch([0.99125949, 1.00632418, 0.99977876, 0.99630269, 0.99267507, 0.99036433, 1.00532789, 0.99156236, 0.99105007, 1.01544229]), target)
target = targets[1]

np.random.seed(0)
assert np.allclose(branch([0.98630028, 1.00067223, 0.98771499, 1.00596817, 0.98722566, 1.00068382, 1.00431318, 1.00589103, 0.99487492, 0.99074395]), target)
target = targets[2]

np.random.seed(0)
assert np.allclose(branch([0.98836047, 0.98309443, 0.99227105, 1.00620314, 0.9898906,  0.98667342, 0.98365538, 0.98601126, 1.00460268, 1.00441958]), target)
