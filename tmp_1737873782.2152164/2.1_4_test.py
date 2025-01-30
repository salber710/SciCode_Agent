from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps



def simulate_light_diffraction(n, d, RL, R0, lambda_):
    '''Function to simulate light diffraction through a lens and plot the intensity distribution.
    Inputs:
    n (float): Refractive index of the lens material, e.g., 1.5062 for K9 glass.
    d (float): Center thickness of the lens in millimeters (mm).
    RL (float): Radius of curvature of the lens in millimeters (mm), positive for convex surfaces.
    R0 (float): Radius of the incident light beam in millimeters (mm).
    lambda_ (float): Wavelength of the incident light in millimeters (mm), e.g., 1.064e-3 mm for infrared light.
    Outputs:
    Ie (numpy.ndarray): A 2D array of intensity values of the diffraction pattern where Ie[i][j] is the ith x and jth y.
    '''
    
    # Constants
    k = 2 * np.pi / lambda_  # Wave number
    f = RL / (n - 1)  # Focal length of the lens

    # Define the discretization grid
    mr2, ne2, mr0 = 51, 61, 81  # Given discretization parameters
    radial_points = np.linspace(-R0, R0, mr2)
    angular_points = np.linspace(0, 2 * np.pi, ne2)

    # Create meshgrid for the radial and angular points
    R, Theta = np.meshgrid(radial_points, angular_points)

    # Calculate the field at each point
    U = np.exp(-(R**2 / R0**2))  # Gaussian beam profile

    # Calculate the phase change introduced by the lens
    lens_phase = np.exp(-1j * k * (R**2 / (2 * f)))

    # Field after the lens
    U_lens = U * lens_phase

    # Use Fourier transform to simulate propagation to the focal plane
    # Fourier transform from spatial domain to frequency domain
    U_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_lens)))

    # Calculate intensity at the focal plane
    Ie = np.abs(U_focal)**2

    # Normalize intensity
    Ie /= np.max(Ie)

    return Ie


try:
    targets = process_hdf5_to_tuple('2.1', 4)
    target = targets[0]
    n=1.5062
    d=3
    RL=0.025e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)

    target = targets[1]
    n=1.5062
    d=4
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)

    target = targets[2]
    n=1.5062
    d=2
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    assert np.allclose(simulate_light_diffraction(n, d, RL, R0, lambda_), target)

    target = targets[3]
    n=1.5062
    d=2
    RL=0.05e3
    R0=1
    lambda_=1.064e-3
    intensity_matrix = simulate_light_diffraction(n, d, RL, R0, lambda_)
    max_index_flat = np.argmax(intensity_matrix)
    assert (max_index_flat==0) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e