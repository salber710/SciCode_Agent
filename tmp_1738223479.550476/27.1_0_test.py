from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np



# Background: In semiconductor physics, the built-in potential (or built-in bias) of a pn-junction is a key concept. It arises due to the difference in Fermi levels between the n-type and p-type regions, which is brought about by the doping concentrations. 
# The intrinsic carrier concentration, n_i, represents the number of electron-hole pairs generated in pure silicon at thermal equilibrium. The doping concentrations, N_A (acceptor concentration in the p-region) 
# and N_D (donor concentration in the n-region), alter the carrier concentrations, leading to a shift in the Fermi levels. 
# The built-in potential is given by the difference in the electrochemical potentials (Fermi levels) across the junction. 
# At room temperature, the thermal voltage (V_T) is approximately 0.0259 V. The Fermi levels φ_p and φ_n can be calculated using the following relationships:
# φ_p = V_T * ln(N_A / n_i) and φ_n = V_T * ln(N_D / n_i).


def Fermi(N_A, N_D, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_A: float, doping concentration in p-type region # cm^{-3}
    N_D: float, doping concentration in n-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''
    
    V_T = 0.0259  # Thermal potential at room temperature in volts
    
    # Calculate the built-in bias for the p-type region
    phi_p = V_T * np.log(N_A / n_i)
    
    # Calculate the built-in bias for the n-type region
    phi_n = V_T * np.log(N_D / n_i)
    
    return phi_p, phi_n


try:
    targets = process_hdf5_to_tuple('27.1', 3)
    target = targets[0]
    assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)

    target = targets[1]
    assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)

    target = targets[2]
    assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e