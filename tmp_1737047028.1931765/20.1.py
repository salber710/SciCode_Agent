import numpy as np



# Background: The Bose-Einstein distribution describes the statistical distribution of identical indistinguishable particles over various energy states in thermal equilibrium. It is particularly applicable to bosons, which are particles that follow Bose-Einstein statistics. The distribution is given by the formula:
# n(ε) = 1 / (exp(ε / (k_B * T)) - 1)
# where n(ε) is the average number of particles with energy ε, k_B is the Boltzmann constant (8.617333262145 x 10^-5 eV/K), and T is the temperature in Kelvin. In this problem, the energy ε is given by the phonon frequency in terahertz (THz), which needs to be converted to electron volts (eV) using the conversion factor 0.004135667 eV/THz. If the temperature is zero, the distribution should return zero, as the formula becomes undefined due to division by zero.


def bose_distribution(freq, temp):
    '''This function defines the bose-einstein distribution
    Input
    freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band
    temp: a float representing the temperature of the distribution
    Output
    nbose: A 2D array of the same shape as freq, representing the Bose-Einstein distribution factor for each frequency.
    '''
    # Boltzmann constant in eV/K
    k_B = 8.617333262145e-5
    # Conversion factor from THz to eV
    thz_to_ev = 0.004135667
    
    # If temperature is zero, return an array of zeros with the same shape as freq
    if temp == 0:
        return np.zeros_like(freq)
    
    # Convert frequencies from THz to eV
    energy = freq * thz_to_ev
    
    # Calculate the Bose-Einstein distribution
    nbose = 1 / (np.exp(energy / (k_B * temp)) - 1)
    
    return nbose


from scicode.parse.parse import process_hdf5_to_tuple

targets = process_hdf5_to_tuple('20.1', 3)
target = targets[0]

assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 0), target)
target = targets[1]

assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 100.0), target)
target = targets[2]

assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 300), target)
