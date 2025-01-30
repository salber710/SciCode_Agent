from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps



# Background: 
# A Gaussian beam is a type of electromagnetic wave that has a Gaussian intensity profile. 
# The beam's field distribution can be described by the Gaussian beam equation, which depends on parameters such as the beam waist (w0), 
# wavelength (Ld), and the distance of propagation (z). 
# The Fourier transform is a mathematical tool that allows us to analyze the frequency components of a signal or function. 
# In optics, the Fourier domain is often used to analyze and propagate beams, as it simplifies the convolution operations involved in wave propagation.
# The Fourier transform of a Gaussian function is also a Gaussian, which makes it particularly convenient for analysis.
# The propagation of a Gaussian beam can be calculated in the Fourier domain and then transformed back to the spatial domain using the inverse Fourier transform.
# The intensity distribution of the beam is given by the absolute square of the field distribution.



def propagate_gaussian_beam(N, Ld, w0, z, L):
    '''Propagate a Gaussian beam and calculate its intensity distribution before and after propagation.
    Input
    N : int
        The number of sampling points in each dimension (assumes a square grid).
    Ld : float
        Wavelength of the Gaussian beam.
    w0 : float
        Waist radius of the Gaussian beam at its narrowest point.
    z : float
        Propagation distance of the Gaussian beam.
    L : float
        Side length of the square area over which the beam is sampled.
    Output
    Gau:     a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution before propagation.
    Gau_Pro: a 2D array with dimensions (N+1, N+1) representing the absolute value of the beam's amplitude distribution after propagation.
    '''

    # Create a grid of spatial coordinates
    x = np.linspace(-L/2, L/2, N+1)
    y = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, y)

    # Calculate the initial Gaussian beam field distribution
    r_squared = X**2 + Y**2
    Gau = np.exp(-r_squared / w0**2)

    # Fourier transform of the initial field
    Gau_ft = np.fft.fftshift(np.fft.fft2(Gau))

    # Frequency coordinates
    fx = np.fft.fftfreq(N+1, d=L/(N+1))
    fy = np.fft.fftfreq(N+1, d=L/(N+1))
    FX, FY = np.meshgrid(fx, fy)

    # Calculate the transfer function for free space propagation
    k = 2 * np.pi / Ld
    H = np.exp(-1j * k * z * np.sqrt(1 - (Ld * FX)**2 - (Ld * FY)**2))

    # Apply the transfer function in the Fourier domain
    Gau_Pro_ft = Gau_ft * H

    # Inverse Fourier transform to get the propagated field
    Gau_Pro = np.fft.ifft2(np.fft.ifftshift(Gau_Pro_ft))

    # Return the absolute value of the field distributions
    return np.abs(Gau), np.abs(Gau_Pro)


try:
    targets = process_hdf5_to_tuple('28.1', 3)
    target = targets[0]
    N = 500  # sample number ,
    L = 10*10**-3   # Full side length 
    Ld = 0.6328 * 10**-6  
    w0 = 1.0 * 10**-3  
    z = 10  
    gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
    # domain specific check 1
    # Physics Calculate the energy info at beginning and end 
    # The result is Intensity distrbution
    # The total energy value is expressed as total power, the intensity integrated over the cross section 
    dx = L/N
    dy = L/N
    dA = dx*dy
    P1 = np.sum(gau1) * dA
    P2 = np.sum(gau2)* dA
    assert np.allclose((P1, P2), target)

    target = targets[1]
    N = 800  
    L = 16*10**-3
    Ld = 0.6328 * 10**-6  
    w0 = 1.5* 10**-3  
    z = 15  
    gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
    # domain specific check 2
    # Physics Calculate the energy info at beginning and end 
    # The result is Intensity distrbution
    # The total energy value is expressed as total power, the intensity integrated over the cross section 
    dx = L/N
    dy = L/N
    dA = dx*dy
    P1 = np.sum(gau1) * dA
    P2 = np.sum(gau2)* dA
    assert np.allclose((P1, P2), target)

    target = targets[2]
    N = 400 
    L = 8*10**-3
    Ld = 0.6328 * 10**-6  
    w0 = 1.5* 10**-3  
    z = 20  
    gau1, gau2= propagate_gaussian_beam(N, Ld, w0, z, L)
    # domain specific 3
    # Physics Calculate the energy info at beginning and end 
    # The result is Intensity distrbution
    # The total energy value is expressed as total power, the intensity integrated over the cross section 
    dx = L/N
    dy = L/N
    dA = dx*dy
    P1 = np.sum(gau1) * dA
    P2 = np.sum(gau2)* dA
    assert np.allclose((P1, P2), target)

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e