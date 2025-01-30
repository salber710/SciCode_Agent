from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from scipy.integrate import simps


def propagate_gaussian_beam(N, Ld, w0, z, L):
    # Create spatial grid
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    
    # Initial Gaussian beam field distribution
    r2 = X**2 + Y**2
    initial_field = np.exp(-r2 / (w0**2))
    
    # Compute the Fourier transform of the initial field
    initial_field_ft = np.fft.fftshift(np.fft.fft2(initial_field))
    
    # Compute frequency coordinates
    dx = x[1] - x[0]
    df = 1 / (N * dx)
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    FX, FY = np.meshgrid(fx, fy)
    
    # Wavenumber
    k = 2 * np.pi / Ld
    
    # Compute the transfer function for Gaussian beam propagation in the Fourier domain
    H = np.exp(-1j * k * z * np.sqrt(1 - (Ld * FX)**2 - (Ld * FY)**2))
    
    # Apply the transfer function to the Fourier transform of the initial field
    propagated_field_ft = initial_field_ft * H
    
    # Compute the inverse Fourier transform to get the propagated field in the spatial domain
    propagated_field = np.fft.ifft2(np.fft.ifftshift(propagated_field_ft))
    
    # Return the absolute values of the initial and propagated fields
    return np.abs(initial_field), np.abs(propagated_field)



def gaussian_beam_through_lens(wavelength, w0, R0, Mf1, z, Mp2, L1, s):
    pi = np.pi
    z_R = pi * w0**2 / wavelength
    q0 = R0 + 1j * z_R

    # Propagate to the lens
    q_lens = (q0 + s) / (1 - s / (R0 + 1j * z_R))

    # Passing through the lens
    q_post_lens = (Mf1[0, 0] * q_lens + Mf1[0, 1]) / (Mf1[1, 0] * q_lens + Mf1[1, 1])

    # Propagation after the lens
    q_final = q_post_lens + L1

    # Calculate waist size at each z position using a direct computation
    wz = np.sqrt(-wavelength / (pi * np.imag(1 / (q_final + z))))

    return wz



# Background: 
# Gaussian beam propagation through optical systems can be analyzed using the ABCD matrix method, which allows us to track the transformation of the beam's complex parameter through lenses and free space. The waist size of the beam changes as it propagates, and the position of the new focus can be determined by analyzing the beam's parameters after passing through the lens. The intensity distribution at the focus is crucial for applications requiring precise beam shaping and focusing. The intensity is proportional to the square of the field amplitude, and the field distribution can be obtained by propagating the beam through free space and optical elements using Fourier optics.



def Gussian_Lens_transmission(N, Ld, z, L, w0, R0, Mf1, Mp2, L1, s):
    # Constants
    pi = np.pi
    k = 2 * pi / Ld  # Wavenumber

    # Initial Rayleigh range
    z_R = pi * w0**2 / Ld

    # Initial complex beam parameter
    q0 = R0 + 1j * z_R

    # Propagate to the lens
    q_lens = (q0 + s) / (1 - s / (R0 + 1j * z_R))

    # Passing through the lens
    q_post_lens = (Mf1[0, 0] * q_lens + Mf1[0, 1]) / (Mf1[1, 0] * q_lens + Mf1[1, 1])

    # Propagation after the lens
    q_final = q_post_lens + L1

    # Calculate waist size at each z position
    Wz = np.sqrt(-Ld / (pi * np.imag(1 / (q_final + z))))

    # Find the new focus position (where the waist is minimum)
    focus_depth = z[np.argmin(Wz)]

    # Calculate the intensity distribution at the new focus plane
    # Create spatial grid
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # Calculate the field distribution at the focus
    r2 = X**2 + Y**2
    waist_at_focus = np.min(Wz)
    field_at_focus = np.exp(-r2 / (waist_at_focus**2))

    # Intensity is the square of the field amplitude
    Intensity = np.abs(field_at_focus)**2

    return Wz, focus_depth, Intensity


try:
    targets = process_hdf5_to_tuple('28.3', 4)
    target = targets[0]
    from scicode.compare.cmp import cmp_tuple_or_list
    Ld = 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 150  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 0
    # User input for beam quality index
    Mp2 = 1
    N = 800  
    L = 16*10**-3
    assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)

    target = targets[1]
    from scicode.compare.cmp import cmp_tuple_or_list
    Ld= 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 180  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 100 # initial waist position
    # User input for beam quality index
    Mp2 = 1.5
    N = 800  
    L = 16*10**-3
    assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)

    target = targets[2]
    from scicode.compare.cmp import cmp_tuple_or_list
    Ld = 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 180  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 150 # initial waist position
    # User input for beam quality index
    Mp2 = 1.5
    N = 800  
    L = 16*10**-3
    assert cmp_tuple_or_list(Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s), target)

    target = targets[3]
    Ld = 1.064e-3  # Wavelength in mm
    w0 = 0.2  # Initial waist size in mm
    R0 = 1.0e30  # Initial curvature radius
    Mf1 = np.array([[1, 0], [-1/50, 1]])  # Free space propagation matrix
    L1 = 180  # Distance in mm
    z = np.linspace(0, 300, 1000)  # Array of distances in mm
    s = 150 # initial waist position
    # User input for beam quality index
    Mp2 = 1.5
    N = 800  
    L = 16*10**-3
    Wz, _, _ = Gussian_Lens_transmission(N, Ld, z, L,w0,R0, Mf1, Mp2, L1, s)
    scalingfactor = max(z)/len(z)
    LensWz = Wz[round(L1/scalingfactor)]
    BeforeLensWz = Wz[round((L1-5)/scalingfactor)]
    AfterLensWz = Wz[round((L1+5)/scalingfactor)]
    assert (LensWz>BeforeLensWz, LensWz>AfterLensWz) == target

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e