import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: In Fourier optics, spatial filtering is a technique used to modify the frequency content of an image.
# A cross-shaped high-pass filter allows high-frequency components to pass through while blocking low-frequency components
# in a specific cross pattern in the Fourier domain. The high-pass filter enhances details in an image by removing 
# low-frequency noise such as background lighting variations. This technique involves transforming the image to the 
# frequency domain using the Fourier Transform, applying a filter mask, and then transforming the filtered image 
# back to the spatial domain using the Inverse Fourier Transform. The filter mask is designed to have zeros (blocking) 
# in the low-frequency regions and ones (passing) in the high-frequency regions, except for the bandwidth around the 
# central cross where frequencies are blocked.

def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float;2D numpy array, the input image.
    bandwidth: bandwidth of cross-shaped filter, int
    Output:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    # Get the dimensions of the input image
    rows, cols = image_array.shape

    # Create a filter mask of the same size as the input image
    T = np.ones((rows, cols), dtype=float)

    # Determine the center of the frequency domain
    center_x, center_y = rows // 2, cols // 2

    # Create the cross-shaped high-pass filter mask
    # Clear out the cross region (horizontal and vertical lines of width 'bandwidth' centered on the middle of the array)
    T[center_x - bandwidth:center_x + bandwidth, :] = 0
    T[:, center_y - bandwidth:center_y + bandwidth] = 0

    # Apply Fourier transform to the image
    F = fftshift(fft2(image_array))

    # Apply the filter mask to the Fourier transformed image
    filtered_F = F * T

    # Transform the filtered image back to the spatial domain
    filtered_image = np.real(ifft2(ifftshift(filtered_F)))

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
