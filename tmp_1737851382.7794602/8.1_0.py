import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: In Fourier optics, spatial filtering is a technique used to modify the frequency components of an image. 
# A high-pass filter allows high-frequency components to pass through while attenuating low-frequency components. 
# A cross-shaped band high-pass filter specifically targets frequencies in a cross pattern, excluding a central bandwidth 
# around the origin in the frequency domain. This is useful for removing low-frequency noise and enhancing high-frequency 
# details. The Fourier Transform is used to convert the image to the frequency domain, where the filter is applied. 
# The inverse Fourier Transform is then used to convert the filtered image back to the spatial domain.



def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float;2D numpy array, the input image.
    bandwidth: bandwidth of cross-shaped filter, int
    Output:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''

    # Get the dimensions of the image
    rows, cols = image_array.shape

    # Create a frequency grid
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Shift the zero frequency component to the center
    U = fftshift(U)
    V = fftshift(V)

    # Create the cross-shaped high-pass filter
    T = np.ones((rows, cols), dtype=float)

    # Apply the bandwidth exclusion in the cross pattern
    T[np.abs(U) < bandwidth] = 0
    T[:, np.abs(V) < bandwidth] = 0

    # Transform the image to the frequency domain
    F_image = fft2(image_array)
    F_image_shifted = fftshift(F_image)

    # Apply the filter in the frequency domain
    F_filtered_shifted = F_image_shifted * T

    # Transform back to the spatial domain
    F_filtered = ifftshift(F_filtered_shifted)
    filtered_image = np.real(ifft2(F_filtered))

    return T, filtered_image

from scicode.parse.parse import process_hdf5_to_tuple
targets = process_hdf5_to_tuple('8.1', 4)
target = targets[0]

from scicode.compare.cmp import cmp_tuple_or_list
matrix = np.array([[1, 0,0,0], [0,0, 0,1]])
bandwidth = 40
image_array = np.tile(matrix, (400, 200))
assert cmp_tuple_or_list(apply_cshband_pass_filter(image_array, bandwidth), target)
target = targets[1]

from scicode.compare.cmp import cmp_tuple_or_list
matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
bandwidth = 40
image_array = np.tile(matrix, (400, 200))
assert cmp_tuple_or_list(apply_cshband_pass_filter(image_array, bandwidth), target)
target = targets[2]

from scicode.compare.cmp import cmp_tuple_or_list
matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
bandwidth = 20
image_array = np.tile(matrix, (400, 200))
assert cmp_tuple_or_list(apply_cshband_pass_filter(image_array, bandwidth), target)
target = targets[3]

matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
bandwidth = 20
image_array = np.tile(matrix, (400, 200))
T1, filtered_image1 = apply_cshband_pass_filter(image_array, bandwidth)
matrix = np.array([[1, 0,1,0], [1,0, 1,0]])
bandwidth = 30
image_array = np.tile(matrix, (400, 200))
T2, filtered_image2 = apply_cshband_pass_filter(image_array, bandwidth)
assert (np.sum(T1)>np.sum(T2)) == target
