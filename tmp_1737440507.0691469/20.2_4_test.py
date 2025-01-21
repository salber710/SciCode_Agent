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



# Background: The phonon angular momentum is calculated using the Bose-Einstein distribution factor and the mode-decomposed phonon angular momentum. 
# The Bose-Einstein distribution factor is given by n(ε) = 1 / (exp(ε/kT) - 1), where ε is the energy of the state, which can be derived from the phonon frequency.
# The mode-decomposed phonon angular momentum is calculated using the polarization vectors of the phonons and involves matrix operations with the Pauli matrices, which represent 
# spin operators. Here, we focus on calculating the z-component of the phonon angular momentum using the Pauli matrix M_z. The formula for the angular momentum is:
# L_z = sum_q_nu [(n_0(omega_q_nu) + 1/2) * l_q_nu^z], where l_q_nu^z = ħ * epsilon_q_nu^† * M_z * epsilon_q_nu.
# The Pauli matrix for the z-component is M_z = [[0, -i, 0], [i, 0, 0], [0, 0, 0]].


def phonon_angular_momentum(freq, polar_vec, temp):
    '''Calculate the phonon angular momentum based on predefined axis orders: alpha=z, beta=x, gamma=y.
       Input
       freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. For example, freq[0][1] is the phonon frequency of the 0th q point on the 1st band
       polar_vec: a numpy array of shape (nqpts, nbnds, natoms, 3) that contains the phonon polarization vectors; each element is a numpy array of 3 complex numbers. 
       nqpts is the number of k points. nbnds is the number of bands. natoms is the number of atoms. For example, polar_vec[0][1][2][:] represents the 1D array of x,y,z components of the 
       polarization vector of the 0th q point of the 1st band of the 2nd atom.
       temp: a float representing the temperature of the distribution in Kelvin
       Output
       momentum: A 3D array containing the mode decomposed phonon angular momentum. The dimension is (3, nqpts, nbnds). For example, momentum[0][1][2] is the x-component
       of the phonon angular momentum of the 1st q point on the 2nd band
       Notes:
       - Angular momentum values are in units of ħ (reduced Planck constant).
    '''
    
    # Boltzmann constant in eV/K
    k_B = 8.617333262145e-5

    # Conversion factor from THz to eV
    thz_to_ev = 0.004135667

    # Pauli matrix for z-component
    M_z = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])

    # Calculate the energies from frequencies
    energies = freq * thz_to_ev

    # Calculate Bose-Einstein distribution factors
    if temp == 0:
        # If temperature is zero, the distribution factors are zero
        bose_factors = np.zeros_like(freq)
    else:
        bose_factors = 1 / (np.exp(energies / (k_B * temp)) - 1)

    # Calculate the phonon angular momentum
    nqpts, nbnds, natoms, _ = polar_vec.shape
    momentum = np.zeros((3, nqpts, nbnds), dtype=np.complex128)

    for q in range(nqpts):
        for nu in range(nbnds):
            # Calculate the mode-decomposed phonon angular momentum for each band and q point
            l_z_q_nu = 0
            for atom in range(natoms):
                epsilon_q_nu = polar_vec[q, nu, atom]
                l_z_q_nu += np.conj(epsilon_q_nu) @ M_z @ epsilon_q_nu
            # L_z = (n + 1/2) * l_z
            momentum[2, q, nu] = (bose_factors[q, nu] + 0.5) * l_z_q_nu

    return momentum

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('20.2', 3)
target = targets[0]

freq = np.array([[1,15]])
polar_vec = np.array ([[[[ 1.35410000e-10+0.00000000e+00j, -5.83670000e-10+0.00000000e+00j,
    -6.33918412e-01+9.17988663e-06j],
   [ 1.35410000e-10+0.00000000e+00j, -5.83670000e-10+0.00000000e+00j,
    -6.33918412e-01+9.17988663e-06j]],
  [[-3.16865726e-01+0.00000000e+00j,  5.48827530e-01-1.00000000e-14j,
    -5.73350000e-10+0.00000000e+00j],
   [-3.16865726e-01+0.00000000e+00j,  5.48827530e-01-1.00000000e-14j,
    -5.73350000e-10+0.00000000e+00j]]]]) 
assert np.allclose(phonon_angular_momentum(freq, polar_vec, 300), target)
target = targets[1]

freq = np.array([[1,2,3]])
polar_vec = np.array ([[[[ 1.91024375e-02+0.00000000e+00j,  2.62257857e-01+0.00000000e+00j,
     0.00000000e+00+0.00000000e+00j],
   [-1.91024375e-02+0.00000000e+00j, -2.62257857e-01+0.00000000e+00j,
     0.00000000e+00+0.00000000e+00j],
   [ 4.76845066e-02+0.00000000e+00j,  6.54661822e-01+0.00000000e+00j,
     0.00000000e+00+0.00000000e+00j]],
  [[-2.71587444e-01+0.00000000e+00j, -1.56801084e-01+0.00000000e+00j,
    -0.00000000e+00+0.00000000e+00j],
   [-2.71587444e-01+0.00000000e+00j, -1.56801084e-01+0.00000000e+00j,
     0.00000000e+00+0.00000000e+00j],
   [ 5.48853587e-01+0.00000000e+00j,  3.16880766e-01+0.00000000e+00j,
     0.00000000e+00+0.00000000e+00j]],
  [[ 0.00000000e+00+0.00000000e+00j, -0.00000000e+00+0.00000000e+00j,
     8.03918626e-02-3.11175396e-03j],
   [ 0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
    -8.03918626e-02+3.11175396e-03j],
   [-0.00000000e+00+0.00000000e+00j,  0.00000000e+00+0.00000000e+00j,
    -7.01989416e-01+2.71721326e-02j]]]])
assert np.allclose(phonon_angular_momentum(freq, polar_vec, 300), target)
target = targets[2]

freq = np.array([[1,2,3]])
polar_vec = np.array ([[[[-3.05049450e-01+0.00000000e+00j, -1.76120375e-01-1.35300000e-11j,
     5.93553844e-03-5.26762876e-01j],
   [-3.04972021e-01-6.87266502e-03j, -1.76075671e-01-3.96793482e-03j,
     5.93376463e-03-5.26762896e-01j],
   [-1.47632691e-01+1.01094998e-03j, -8.52357703e-02+5.83672318e-04j,
     2.94736973e-03-2.63327725e-01j]],
    [[ 3.16888651e-01+0.00000000e+00j, -5.48867245e-01-5.40280000e-10j,
    -8.09100000e-11-7.36970000e-09j],
   [ 3.16814487e-01-6.85546961e-03j, -5.48738790e-01+1.18740222e-02j,
    -7.89300000e-11-7.36972000e-09j],
   [ 1.56780446e-01+1.21424995e-03j, -2.71551699e-01-2.10314271e-03j,
    -4.00600000e-11-3.68774000e-09j]],
    [[-4.56339422e-01+0.00000000e+00j, -2.63467692e-01-3.69580000e-10j,
    -3.75795954e-03+3.52735968e-01j],
   [-4.56236041e-01-9.71303118e-03j, -2.63408005e-01-5.60782088e-03j,
    -3.75075830e-03+3.52736045e-01j],
   [-2.27797069e-01+1.84208194e-03j, -1.31518701e-01+1.06352639e-03j,
    -1.83538790e-03+1.69427730e-01j]]]])
assert np.allclose(phonon_angular_momentum(freq, polar_vec, 300), target)
