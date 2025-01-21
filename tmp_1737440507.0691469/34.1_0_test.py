import numpy as np



# Background: 
# In semiconductor physics, the Fermi level represents the energy level at which the probability of being occupied by an electron is 50%. For doped semiconductors, the position of the Fermi level changes relative to the intrinsic energy level (E_i) depending on the type and concentration of doping. The built-in potential (or built-in bias) for the p-type (phi_p) and n-type (phi_n) regions can be calculated using the doping concentrations and the intrinsic carrier density.
# The built-in potential for p-type (phi_p) is calculated as:
# phi_p = (kT/q) * ln(N_a / n_i)
# The built-in potential for n-type (phi_n) is calculated as:
# phi_n = (kT/q) * ln(N_d / n_i)
# Here, kT/q is the thermal voltage at room temperature, which is approximately 0.0259V.


def Fermi(N_a, N_d, n_i):
    '''This function computes the Fermi levels of the n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    Outputs:
    phi_p: float, built-in bias in p-type region (compare to E_i)
    phi_n: float, built-in bias in n-type region (compare to E_i)
    '''

    # Thermal potential at room temperature
    Vt = 0.0259  # volts

    # Calculate the built-in bias for p-type region (phi_p)
    phi_p = Vt * np.log(N_a / n_i)

    # Calculate the built-in bias for n-type region (phi_n)
    phi_n = Vt * np.log(N_d / n_i)

    return phi_p, phi_n

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('34.1', 3)
target = targets[0]

assert np.allclose(Fermi(2*10**17,3*10**17,10**12), target)
target = targets[1]

assert np.allclose(Fermi(1*10**17,2*10**17,10**12), target)
target = targets[2]

assert np.allclose(Fermi(2*10**17,3*10**17,2*10**11), target)
