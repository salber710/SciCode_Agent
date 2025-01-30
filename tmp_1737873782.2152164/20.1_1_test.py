from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np




def bose_distribution(freq, temp):
    '''This function defines the bose-einstein distribution
    Input
    freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band
    temp: a float representing the temperature of the distribution
    Output
    nbose: A 2D array of the same shape as freq, representing the Bose-Einstein distribution factor for each frequency.
    '''
    
    # Conversion factor from THz to eV
    conversion_factor = 0.004135667
    
    # If temperature is zero, Bose-Einstein distribution is zero.
    if temp == 0:
        return np.zeros_like(freq)
    
    # Calculate the Bose-Einstein distribution
    # Using the formula: n(omega) = 1 / (exp(h * omega / (k_B * T)) - 1)
    # where omega is the frequency in eV (converted from THz)
    # h is Planck's constant, k_B is Boltzmann's constant
    # Here, we consider the conversion factor to get omega in eV
    omega_in_eV = freq * conversion_factor
    
    # Boltzmann's constant in eV/K
    k_B = 8.617333262145e-5
    
    # Calculating the Bose-Einstein distribution factor
    nbose = 1.0 / (np.exp(omega_in_eV / (k_B * temp)) - 1.0)
    
    return nbose


try:
    targets = process_hdf5_to_tuple('20.1', 3)
    target = targets[0]
    assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 0), target)

    target = targets[1]
    assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 100.0), target)

    target = targets[2]
    assert np.allclose(bose_distribution(np.array([[1,2],[3,4]]), 300), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e