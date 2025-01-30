from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

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
    V_T = 0.0259

    # Calculate the natural logarithm using the Newton-Raphson method
    def newton_raphson_ln(x, tolerance=1e-10, max_iterations=100):
        if x <= 0:
            raise ValueError("x must be positive for logarithm")
        guess = x - 1.0
        for _ in range(max_iterations):
            next_guess = guess + 2 * ((x - pow(2, guess)) / (x + pow(2, guess)))
            if abs(next_guess - guess) < tolerance:
                return next_guess
            guess = next_guess
        return guess

    # Calculate phi_p and phi_n using the Newton-Raphson method for natural logarithm
    phi_p = V_T * newton_raphson_ln(N_a / n_i)
    phi_n = V_T * newton_raphson_ln(N_d / n_i)

    return phi_p, phi_n


def depletion(N_a, N_d, n_i, e_r):
    '''This function calculates the depletion width in both n-type and p-type regions.
    Inputs:
    N_d: float, doping concentration in n-type region # cm^{-3}
    N_a: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    '''

    # Constants
    e_0 = 8.854e-14  # Vacuum permittivity in F/cm
    q = 1.6e-19      # Electron charge in C

    # Calculate the built-in potential using the Fermi function
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    phi_total = phi_p + phi_n

    # Calculate the depletion widths using an energy balance approach
    # This method involves balancing the charge densities and potential energy
    energy_balance_factor = (2 * e_r * e_0 * phi_total) / q
    charge_density_product = (N_a * N_d) ** 0.5

    xn = (energy_balance_factor * N_a / (N_a + N_d) / charge_density_product) ** 0.5
    xp = (energy_balance_factor * N_d / (N_a + N_d) / charge_density_product) ** 0.5

    return xn, xp




def potential(N_a, N_d, n_i, e_r):
    '''Inputs:
    N_a: float, doping concentration in n-type region # cm^{-3}
    N_d: float, doping concentration in p-type region # cm^{-3}
    n_i: float, intrinsic carrier density # cm^{-3}
    e_r: float, relative permittivity
    Outputs:
    xn: float, depletion width in n-type side # cm
    xp: float, depletion width in p-type side # cm
    potential: np.array, the potential distribution
    '''
    
    # Use the previous functions to get depletion widths
    xn, xp = depletion(N_a, N_d, n_i, e_r)
    
    # Total depletion width
    x_total = xn + xp
    
    # Space increment in cm (0.1 nm = 1e-8 cm)
    dx = 1e-8
    
    # Electron charge
    q = 1.6e-19  # C
    
    # Vacuum permittivity
    e_0 = 8.854e-14  # F/cm
    
    # Calculate the built-in potential using the Fermi function
    phi_p, phi_n = Fermi(N_a, N_d, n_i)
    phi_total = phi_p + phi_n
    
    # Create potential array with an oscillatory potential distribution
    x_positions = np.arange(0, x_total, dx)
    potential = np.zeros_like(x_positions)
    
    # Calculate potential based on position
    for i, x in enumerate(x_positions):
        if x <= xp:  # In the p-type region
            # Oscillatory potential variation
            potential[i] = phi_p * (0.5 + 0.5 * np.sin(2 * np.pi * x / xp))
        else:  # In the n-type region
            # Oscillatory potential variation
            x_rel = x - xp
            potential[i] = phi_p + phi_n * (0.5 + 0.5 * np.sin(2 * np.pi * x_rel / xn))
    
    return xn, xp, potential


try:
    targets = process_hdf5_to_tuple('34.3', 4)
    target = targets[0]
    xn,xp,_ = potential(2*10**17,2*10**17,10**11,15)
    assert (xn==xp) == target

    target = targets[1]
    assert np.allclose(potential(1*10**18,2*10**18,10**11,15)[2], target)

    target = targets[2]
    assert np.allclose(potential(1*10**17,2*10**17,10**11,15)[2], target)

    target = targets[3]
    assert np.allclose(potential(1*10**18,2*10**17,10**11,10)[2], target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e