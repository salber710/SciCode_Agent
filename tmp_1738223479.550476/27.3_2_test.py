from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np

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

    # Thermal potential at room temperature in volts
    V_T = 0.0259

    # Calculate built-in biases using Simpson's Rule for integral approximation
    def simpsons_log_approximation(x, n_segments=1000):
        """Approximate natural log(x) using Simpson's Rule."""
        if x <= 0:
            raise ValueError("Logarithm input must be positive")
        
        f = lambda t: 1 / t
        a, b = 1, x
        h = (b - a) / n_segments
        integral = f(a) + f(b)
        
        for i in range(1, n_segments, 2):
            integral += 4 * f(a + i * h)
        for i in range(2, n_segments-1, 2):
            integral += 2 * f(a + i * h)
        
        integral *= h / 3
        return integral

    # Calculate the built-in biases using the Simpson's Rule approximation
    phi_p = V_T * simpsons_log_approximation(N_A / n_i)
    phi_n = V_T * simpsons_log_approximation(N_D / n_i)

    return phi_p, phi_n


def capacitance(xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the capacitance of a p-i-n diode using a unique method.
    Input:
    xi (float): Width of the intrinsic region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density of the material # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the p-i-n diode (V).
    Output:
    C (float): Capacitance of the p-i-n diode (F).
    '''


    # Constants
    epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
    e_charge = 1.602e-19   # Electron charge in C

    # Convert intrinsic layer thickness and area from micrometers to meters
    xi_m = xi * 1e-6
    A_m2 = A * 1e-12

    # Built-in potential using a different logarithmic form
    V_bi = 0.026 * (np.log(N_A / n_i) + np.log(N_D / n_i))

    # Total voltage considering the built-in potential
    V_total = V_bi + V0

    # Depletion width using a unique formula
    W_eff = np.sqrt((2 * es * epsilon_0 * V_total) / (e_charge * (N_A * N_D) / (N_A + N_D)))

    # Capacitance calculation using a different effective width adjustment
    C = (es * epsilon_0 * A_m2) / (xi_m + W_eff / 7)

    return C



def get_3dB_frequency(R, xi, A, N_A, N_D, n_i, es, V0):
    '''Calculates the 3dB frequency of a photodetector.
    Input:
    R (float): Load resistance (Ohms).
    xi (float): Intrinsic width of the depletion region (μm).
    A (float): Detector Area (μm^2).
    N_A (float): Doping concentration of the p-type region (cm^-3).
    N_D (float): Doping concentration of the n-type region (cm^-3).
    n_i: float, intrinsic carrier density # cm^{-3}
    es (float): Relative permittivity.
    V0 (float): Applied voltage to the PN junction (V).
    Output:
    f_3dB (float): 3dB frequency (Hz).
    '''

    # Step 1: Calculate the capacitance using the provided function
    C = capacitance(xi, A, N_A, N_D, n_i, es, V0)

    # Step 2: Calculate the RC time constant
    RC_time_constant = R * C

    # Step 3: Derive the 3dB frequency using an alternative approach
    # Instead of direct division, use exponentiation
    f_3dB = 1 / (2 * 3.14159265 * RC_time_constant)

    return f_3dB


try:
    targets = process_hdf5_to_tuple('27.3', 4)
    target = targets[0]
    xi_arr = np.linspace(0, 5, 50)
    f3dB = get_3dB_frequency(50, xi_arr, 700, 1e19, 1e17, 1.8e6, 13, 0)
    xi_test = np.linspace(0, 5, 50)
    f_test = get_3dB_frequency(50, xi_test, 700, 1e50, 1e50, 1.8e6, 13, 0)
    score = (f3dB - f_test)/f3dB
    assert (np.min(score)==score[-1] and np.max(score)==score[0]) == target

    target = targets[1]
    xi_arr = np.linspace(0, 5, 50)
    assert np.allclose(get_3dB_frequency(50, xi_arr, 1400, 1e16, 1e15,1.8e6,13, 0), target)

    target = targets[2]
    xi_arr = np.linspace(0, 5, 50)
    assert np.allclose(get_3dB_frequency(50, xi_arr, 1400, 1e19, 1e17,1.8e6,13, 0), target)

    target = targets[3]
    xi_arr = np.linspace(0, 5, 50)
    assert np.allclose(get_3dB_frequency(50, xi_arr, 5000, 1e19, 1e17,1.8e6,13, 0), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e