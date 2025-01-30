from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
import itertools



def ground_state_wavelength(L, mr):
    '''Given the width of a infinite square well, provide the corresponding wavelength of the ground state eigen-state energy.
    Input:
    L (float): Width of the infinite square well (nm).
    mr (float): relative effective electron mass.
    Output:
    lmbd (float): Wavelength of the ground state energy (nm).
    '''
    
    # Constants
    h = 6.626e-34  # Planck constant in J*s
    c = 3e8  # Speed of light in m/s
    me = 9.109e-31  # Free electron mass in kg
    
    # Convert L from nanometers to meters
    L_m = L * 1e-9
    
    # Calculate the ground state energy E1
    # E1 = (pi^2 * h^2) / (2 * m * L^2)
    # where m = mr * me
    m = mr * me
    E1 = (np.pi**2 * h**2) / (2 * m * L_m**2)
    
    # Calculate the corresponding photon wavelength lambda
    # E = hc / lambda => lambda = hc / E
    lmbd = (h * c) / E1
    
    # Convert lambda from meters to nanometers
    lmbd_nm = lmbd * 1e9
    
    return lmbd_nm


try:
    targets = process_hdf5_to_tuple('35.1', 3)
    target = targets[0]
    assert np.allclose(ground_state_wavelength(5,0.6), target)

    target = targets[1]
    assert np.allclose(ground_state_wavelength(10,0.6), target)

    target = targets[2]
    assert np.allclose(ground_state_wavelength(10,0.06), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e