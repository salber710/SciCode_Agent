from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: 
# In Fourier optics, spatial filtering is a technique used to modify the frequency content of an image by transforming the image into the frequency domain using the Fourier Transform.
# A high pass filter allows higher frequencies to pass through while blocking lower frequencies. The cross-shaped band high pass filter is designed by specifying a bandwidth, which determines the width of the cross.
# This filter is implemented in the frequency domain by creating a mask that blocks frequencies within the specified bandwidth while allowing those outside of it.
# The central part of the frequency domain image corresponds to low frequencies, and the outer parts correspond to high frequencies. The filter mask will be applied on this frequency domain representation of the image.
# After applying the filter, the image is transformed back to the spatial domain using the inverse Fourier Transform.

def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float; 2D numpy array, the input image.
    bandwidth: bandwidth of cross-shaped filter, int
    Output:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''



    # Get the size of the image
    rows, cols = image_array.shape

    # Create a filter mask initialized to ones
    T = np.ones((rows, cols), dtype=float)

    # Calculate the center of the frequency domain image
    center_row, center_col = rows // 2, cols // 2

    # Create the cross-shaped band high pass filter
    # Zero out the cross-shaped region in the mask
    # Horizontal band
    T[center_row - bandwidth:center_row + bandwidth + 1, :] = 0
    # Vertical band
    T[:, center_col - bandwidth:center_col + bandwidth + 1] = 0

    # Transform the image to the frequency domain
    image_fft = fft2(image_array)
    image_fft_shifted = fftshift(image_fft)

    # Apply the filter mask in the frequency domain
    filtered_fft_shifted = image_fft_shifted * T

    # Transform back to the spatial domain
    filtered_fft = ifftshift(filtered_fft_shifted)
    filtered_image = np.real(ifft2(filtered_fft))

    return T, filtered_image


try:
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

