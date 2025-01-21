import numpy as np



# Background: The Bose-Einstein distribution describes the statistical distribution of identical indistinguishable particles over various energy states in thermal equilibrium. 
# It is particularly useful for understanding the behavior of bosons, such as phonons, at different temperatures. 
# The distribution is given by the formula: n(ε) = 1 / (exp(ε/kT) - 1), where ε is the energy of the state, k is the Boltzmann constant (in eV/K), and T is the temperature in Kelvin.
# For phonons, the energy ε can be related to frequency via the conversion factor from terahertz (THz) to electron volts (eV): ε = freq * 0.004135667.
# At absolute zero temperature, T = 0, the distribution is defined to be zero to avoid division by zero.


def bose_distribution(freq, temp):
    '''This function defines the Bose-Einstein distribution
    Input
    freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. 
          For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band.
    temp: a float representing the temperature of the distribution.
    Output
    nbose: A 2D array of the same shape as freq, representing the Bose-Einstein distribution factor for each frequency.
    '''

    # Boltzmann constant in eV/K
    k_B = 8.617333262145e-5

    # Convert freq from THz to eV
    energy = freq * 0.004135667

    # Initialize the output array with the same shape as freq
    nbose = np.zeros_like(freq)

    if temp == 0:
        # If the temperature is zero, all distribution factors are zero
        return nbose

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
