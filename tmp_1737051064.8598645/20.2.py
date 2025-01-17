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



# Background: The phonon angular momentum is calculated using the Bose-Einstein distribution and the mode-decomposed phonon angular momentum. 
# The Bose-Einstein distribution, n(ε), describes the average number of phonons at a given energy state and temperature. 
# The mode-decomposed phonon angular momentum, l_{q,ν}^α, is calculated using the phonon polarization vectors and a matrix M_α, 
# which represents the angular momentum operator for a given axis α. In this problem, we focus on the z-component of the angular momentum.
# The formula for the phonon angular momentum is:
# L_α^ph = Σ_{q,ν} [n_0(ω_{q,ν}) + 1/2] l_{q,ν}^α
# where n_0(ω_{q,ν}) is the Bose-Einstein distribution factor, and l_{q,ν}^α is given by:
# l_{q,ν}^α = ħ ε_{q,ν}^† M_α ε_{q,ν}
# Here, ε_{q,ν} are the phonon polarization vectors, and M_α is the angular momentum operator matrix for axis α.


def phonon_angular_momentum(freq, polar_vec, temp):
    '''Calculate the phonon angular momentum based on predefined axis orders: alpha=z, beta=x, gamma=y.
    Input
    freq: a 2D numpy array of dimension (nqpts, nbnds) that contains the phonon frequencies; each element is a float. 
    polar_vec: a numpy array of shape (nqpts, nbnds, natoms, 3) that contains the phonon polarization vectors; each element is a numpy array of 3 complex numbers. 
    temp: a float representing the temperature of the distribution in Kelvin
    Output
    momentum: A 3D array containing the mode decomposed phonon angular momentum. The dimension is (3, nqpts, nbnds).
    Notes:
    - Angular momentum values are in units of ħ (reduced Planck constant).
    '''
    
    # Constants
    hbar = 1.0545718e-34  # Planck constant over 2π in J·s
    thz_to_ev = 0.004135667  # Conversion factor from THz to eV
    ev_to_joule = 1.60218e-19  # Conversion factor from eV to Joules
    k_B = 8.617333262145e-5  # Boltzmann constant in eV/K

    # Convert frequencies from THz to eV
    energy = freq * thz_to_ev

    # Calculate the Bose-Einstein distribution
    if temp == 0:
        n_bose = np.zeros_like(freq)
    else:
        n_bose = 1 / (np.exp(energy / (k_B * temp)) - 1)

    # Add 1/2 to the Bose-Einstein distribution factor
    n_bose_plus_half = n_bose + 0.5

    # Define the angular momentum operator matrices for x, y, z
    M_x = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
    M_y = np.array([[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]])
    M_z = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])

    # Initialize the momentum array
    nqpts, nbnds, natoms, _ = polar_vec.shape
    momentum = np.zeros((3, nqpts, nbnds), dtype=np.complex128)

    # Calculate the phonon angular momentum
    for q in range(nqpts):
        for nu in range(nbnds):
            for alpha, M_alpha in enumerate([M_z, M_x, M_y]):
                # Calculate l_{q,ν}^α = ħ ε_{q,ν}^† M_α ε_{q,ν}
                epsilon = polar_vec[q, nu]
                l_q_nu_alpha = hbar * np.einsum('ai,ij,aj->', epsilon.conj(), M_alpha, epsilon)
                # Sum over all q and ν
                momentum[alpha, q, nu] = n_bose_plus_half[q, nu] * l_q_nu_alpha

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
