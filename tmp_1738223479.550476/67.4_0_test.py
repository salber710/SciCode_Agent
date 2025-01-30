from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

def f_V(q, d, bg_eps, l1, l2):
    '''Write down the form factor f(q;l1,l2)
    Input
    q: in-plane momentum, float in the unit of inverse angstrom
    d: layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float, dimensionless
    l1, l2: layer number where z = l*d, integer
    Output
    form_factor: form factor, float
    '''


    # Compute the z-coordinates for the two layers
    z1 = l1 * d
    z2 = l2 * d

    # Calculate the distance in the z-direction
    delta_z = abs(z2 - z1)

    # Implement a form factor using an inverse hyperbolic cosine function
    # This introduces a unique decay based on q and delta_z
    form_factor = np.arccosh(1 + q * delta_z) / (1 + np.sinh(q * d)) / bg_eps

    return form_factor




def D_2DEG(q, omega, gamma, n_eff, e_F, k_F, v_F):
    '''Write down the exact form of density-density correlation function
    Input
    q, in-plane momentum, float in the unit of inverse angstrom
    omega, energy, real part, float in the unit of meV
    gamma, energy, imaginary part, float in the unit of meV
    n_eff, electron density, float in the unit of per square angstrom
    e_F, Fermi energy, float in the unit of meV
    k_F, Fermi momentum, float in the unit of inverse angstrom
    v_F, hbar * Fermi velocity, float in the unit of meV times angstrom
    Output
    D0: density-density correlation function, complex array in the unit of per square angstrom per meV
    '''

    # Compute the complex frequency ω + iγ
    complex_omega = omega + 1j * gamma

    # Convert q to a dimensionless parameter
    q_dimensionless = q / k_F

    # Distinct approach using residue theorem and poles analysis
    def residue_response(q_dimensionless, complex_omega):
        # Calculate poles in the complex plane
        def pole_integrand(k):
            # Energy dispersion and complex frequency shift
            epsilon_k = k**2 / (2 * e_F)
            delta_e = epsilon_k + complex_omega - e_F

            # Calculate residues at the poles
            if abs(delta_e) < 1e-10:
                # Avoid division by zero, return zero contribution for very small delta_e
                return 0
            else:
                return (k / (2 * np.pi)) * np.log(abs((delta_e + q_dimensionless * v_F) / (delta_e - q_dimensionless * v_F)))

        # Define integration limits over a relevant range of the Fermi surface
        k_min, k_max = 0, 2 * k_F

        # Perform integration over the poles
        real_integral, real_error = quad(lambda k: pole_integrand(k).real, k_min, k_max)
        imag_integral, imag_error = quad(lambda k: pole_integrand(k).imag, k_min, k_max)

        # Return the density-density correlation function
        return (n_eff / e_F) * (real_integral + 1j * imag_integral)

    # Calculate the density-density correlation function
    D0 = residue_response(q_dimensionless, complex_omega)

    return D0



def D_cal(D0, q, d, bg_eps, N):
    '''Calculate the matrix form of density-density correlation function D(l1,l2)
    Input
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    bg_eps: LEG dielectric constant, float
    N: matrix dimension, integer
    Output
    D: NxN complex matrix, in the unit of per square angstrom per meV
    '''
    
    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^{-1} μm^{-1}
    
    # Calculate the prefactor for the Coulomb interaction
    V_q = 2 * np.pi * epsilon_0 / (q * bg_eps)
    
    # Initialize the Coulomb interaction matrix V with a Bessel function-based decay
    V = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            distance = np.abs(i - j) * d
            if distance != 0:
                V[i, j] = V_q * np.i0(q * distance) / bg_eps  # Using modified Bessel function of the first kind
            else:
                V[i, j] = V_q  # Handle the singularity at zero distance
    
    # Construct the dielectric matrix ε = I - V * D0 with a different order of operations
    identity_matrix = np.eye(N, dtype=complex)
    epsilon_matrix = identity_matrix - np.dot(V, D0)
    
    # Use the iterative GMRES method to solve ε * D = D0 for D

    D, _ = gmres(epsilon_matrix, D0.flatten())
    D = D.reshape((N, N))
    
    return D



# Background: In a layered electron gas (LEG), the density-density correlation function captures 
# the response of the electron gas to external perturbations. For a bulk system with translational 
# symmetry along the z-axis, this function can be analyzed using the Random Phase Approximation (RPA). 
# The out-of-plane momentum qz arises from the discrete Fourier transform of spatial correlations 
# along the z-direction. In RPA, the dielectric function modifies the bare Coulomb interaction, leading
# to an effective interaction encapsulated by the density-density correlation function D_b(qz).
# The vacuum dielectric constant is given as ε_0 = 55.26349406 e^2 eV^{-1} μm^{-1}, and the LEG dielectric 
# constant is bg_eps. The in-plane momentum q and layer spacing d are critical parameters that affect 
# the interaction strength and decay along the layers.

def D_b_qz_analy(qz, D0, bg_eps, q, d):
    '''Calculate the explicit form of density-density correlation function D_b(qz)
    Input
    qz, out-of-plane momentum, float in the unit of inverse angstrom
    D0, density-density correlation function, complex array in the unit of per square angstrom per meV
    bg_eps: LEG dielectric constant, float
    q, in-plane momentum, float in the unit of inverse angstrom
    d, layer spacing, float in the unit of angstrom
    Output
    D_b_qz: density-density correlation function, complex array in the unit of per square angstrom per meV
    '''
    # Vacuum dielectric constant
    epsilon_0 = 55.26349406  # e^2 eV^{-1} μm^{-1}

    # Calculate the prefactor for the Coulomb interaction using the in-plane momentum q
    V_q = (2 * np.pi * epsilon_0) / (q * bg_eps)

    # Compute the modified Coulomb interaction for bulk system including out-of-plane momentum qz
    # The interaction includes a phase factor exp(-i * qz * d) due to the layered structure
    V_qz = V_q / (1 + V_q * D0 * (1 - np.exp(-1j * qz * d)))

    # Calculate the density-density correlation function D_b(qz) using RPA
    # This involves the response function D0 and the interaction V_qz
    D_b_qz = D0 / (1 - V_qz * D0)

    return D_b_qz


try:
    targets = process_hdf5_to_tuple('67.4', 3)
    target = targets[0]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 2*e_F
    gamma = 0.3
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    qz = -1*np.pi/d
    assert np.allclose(D_b_qz_analy(qz,D0,bg_eps,q,d), target, atol=1e-10, rtol=1e-10)

    target = targets[1]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 2*e_F
    gamma = 0.3
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    qz = 0.2*np.pi/d
    assert np.allclose(D_b_qz_analy(qz,D0,bg_eps,q,d), target, atol=1e-10, rtol=1e-10)

    target = targets[2]
    n_eff = 7.3*10**11 *10**-16   ###unit: A^-2
    m_eff = 0.07 ###unit: m_e (electron mass)
    e_F = 10**3 * np.pi * (7.619964231070681/m_eff) * n_eff   ###Fermi energy, unit: meV
    k_F = np.sqrt(2*np.pi*n_eff)   ###Fermi momentum, unit: A-1
    v_F = 10**3 * (7.619964231070681/m_eff) * k_F   ###hbar * Fermi velocity, unit: meV A
    d = 890   ### unit: A
    bg_eps = 13.1
    q = 0.1*k_F
    omega = 2*e_F
    gamma = 0.3
    D0 = D_2DEG(q,omega,gamma,n_eff,e_F,k_F,v_F)
    qz = 1*np.pi/d
    assert np.allclose(D_b_qz_analy(qz,D0,bg_eps,q,d), target, atol=1e-10, rtol=1e-10)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e