import numpy as np



# Background: The Bose-Einstein distribution describes the statistical distribution of identical indistinguishable bosons over energy states in thermal equilibrium. 
# It is given by the formula: n(ε) = 1 / (exp(ε / (k_B * T)) - 1), where ε is the energy of the state, k_B is the Boltzmann constant, and T is the temperature.
# In this context, the energy ε is given in terms of phonon frequency (in THz), which needs to be converted to energy in electron volts (eV) using the conversion factor 0.004135667 eV/THz.
# If the temperature T is zero, the distribution should return zero, as the formula becomes undefined (division by zero in the exponent).


def bose_distribution(freq, temp):
    '''This function defines the bose-einstein distribution
    Input
    freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band
    temp: a float representing the temperature of the distribution
    Output
    nbose: A 2D array of the same shape as freq, representing the Bose-Einstein distribution factor for each frequency.
    '''
    # Constants
    THZ_TO_EV = 0.004135667  # Conversion factor from THz to eV
    K_B = 8.617333262145e-5  # Boltzmann constant in eV/K

    # If temperature is zero, return an array of zeros with the same shape as freq
    if temp == 0:
        return np.zeros_like(freq)

    # Calculate the Bose-Einstein distribution
    energy = freq * THZ_TO_EV  # Convert frequency to energy in eV
    nbose = 1 / (np.exp(energy / (K_B * temp)) - 1)

    return nbose

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('20.1', 3)
target = targets[0]

assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 0), target)
target = targets[1]

assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 100.0), target)
target = targets[2]

assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 300), target)
