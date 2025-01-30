from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps




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

    # Create spatial coordinate grid
    x = np.linspace(-L/2, L/2, N+1)
    y = np.linspace(-L/2, L/2, N+1)
    X, Y = np.meshgrid(x, y)
    r_squared = X**2 + Y**2

    # Calculate Rayleigh range
    zR = np.pi * w0**2 / Ld

    # Calculate beam waist at distance z
    wz = w0 * np.sqrt(1 + (z/zR)**2)

    # Calculate Gaussian beam distribution before propagation
    Gau = np.exp(-r_squared / w0**2)

    # Fourier domain calculations for propagation
    k = 2 * np.pi / Ld
    fx = np.fft.fftfreq(N+1, L/N)
    fy = np.fft.fftfreq(N+1, L/N)
    Fx, Fy = np.meshgrid(fx, fy)
    F_squared = Fx**2 + Fy**2

    # Propagation in Fourier domain
    H = np.exp(-1j * np.pi * Ld * z * F_squared)
    Gau_fft = np.fft.fft2(Gau)
    Gau_pro_fft = Gau_fft * H
    Gau_pro = np.fft.ifft2(Gau_pro_fft)

    # Return the absolute value of the amplitude distributions
    return np.abs(Gau), np.abs(Gau_pro)


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