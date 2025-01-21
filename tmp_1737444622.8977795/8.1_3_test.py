import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: Fourier optics is a field of optics that studies the way light can be manipulated in terms of its frequency components.
# A spatial filter in the context of Fourier optics is used to manipulate the spatial frequency content of an optical image.
# High-pass filters allow high-frequency components to pass through while attenuating low-frequency components, thereby removing
# the smooth, slowly-varying parts of an image. A cross-shaped band high-pass spatial filter specifically targets frequencies
# along the horizontal and vertical axes, forming a cross pattern in the frequency domain. This can be useful for emphasizing
# features that are oriented horizontally or vertically in an image while suppressing noise and unwanted patterns that occur
# at lower frequencies. The bandwidth parameter determines the width of the cross in the frequency domain, excluding these
# central frequencies from the filtered image.

def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float; 2D numpy array, the input image.
    bandwidth: bandwidth of cross-shaped filter, int
    Output:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    
    # Get the dimensions of the input image
    rows, cols = image_array.shape
    
    # Compute the 2D FFT of the image
    fft_image = fft2(image_array)
    fft_image_shifted = fftshift(fft_image)
    
    # Prepare the cross-shaped high-pass filter
    T = np.ones((rows, cols), dtype=float)  # Initialize the filter as all-pass (ones)
    
    # Calculate center of the frequency domain
    center_row, center_col = rows // 2, cols // 2
    
    # Create cross-shaped filter by setting a band of frequencies around the center to zero
    T[center_row - bandwidth:center_row + bandwidth, :] = 0  # Zero out a horizontal band
    T[:, center_col - bandwidth:center_col + bandwidth] = 0  # Zero out a vertical band
    
    # Apply the filter to the shifted FFT image
    filtered_fft_shifted = fft_image_shifted * T
    
    # Inverse shift the FFT result
    filtered_fft = ifftshift(filtered_fft_shifted)
    
    # Compute the inverse FFT to get the filtered image in the spatial domain
    filtered_image = np.real(ifft2(filtered_fft))
    
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
