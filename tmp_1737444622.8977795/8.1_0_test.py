import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: 
# In Fourier optics, spatial filtering involves transforming an image into its frequency domain using the Fourier transform.
# A cross-shaped high-pass filter in the frequency domain allows frequencies outside a specified bandwidth to pass while blocking 
# those within the bandwidth along the cross shape (both horizontal and vertical axes). This technique helps in removing low-frequency 
# components which often correspond to background noise or unwanted patterns, thereby enhancing the details in the image.
# The Fourier transform facilitates the transition from space (image) to frequency domain, and the inverse Fourier transform 
# brings the image back to its spatial domain after filtering.



def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float;2D numpy array, the input image.
    bandwidth: bandwidth of cross-shaped filter, int
    Ouput:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    
    # Get the shape of the image
    rows, cols = image_array.shape
    
    # Fourier transform of the image
    frequency_image = fft2(image_array)
    
    # Shift the zero frequency component to the center
    shifted_frequency_image = fftshift(frequency_image)
    
    # Create the cross-shaped high-pass filter mask
    T = np.ones((rows, cols), dtype=float)
    
    # Block the low frequencies in the horizontal and vertical center lines
    center_row, center_col = rows // 2, cols // 2
    
    # Create horizontal block
    T[center_row-bandwidth:center_row+bandwidth, :] = 0
    
    # Create vertical block
    T[:, center_col-bandwidth:center_col+bandwidth] = 0
    
    # Apply the filter in the frequency domain
    filtered_frequency = shifted_frequency_image * T
    
    # Shift back the zero frequency component to the original position
    filtered_frequency = ifftshift(filtered_frequency)
    
    # Inverse Fourier transform to get the filtered image back in spatial domain
    filtered_image = ifft2(filtered_frequency)
    
    # Since the result might have small imaginary parts due to numerical errors, take the real part
    filtered_image = np.real(filtered_image)
    
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
