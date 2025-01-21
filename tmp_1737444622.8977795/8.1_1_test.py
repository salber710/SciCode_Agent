import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: In Fourier optics, spatial filtering is a process that alters the spatial frequency components of an image. 
# High-pass filters are used to remove low-frequency components, allowing high-frequency components to pass, which are 
# typically associated with sharp edges and fine details in an image. A cross-shaped band high-pass filter specifically 
# removes frequencies within a cross-shaped band in the frequency domain, leaving out the central low-frequency region 
# and a cross-shaped high-frequency region. The bandwidth of this filter determines how wide the cross-band is. 
# By applying such a filter in the frequency domain and then transforming back to the spatial domain, we can enhance 
# or suppress certain features in the image.



def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float; 2D numpy array, the input image.
    bandwidth: bandwidth of cross-shaped filter, int
    Output:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    
    # Get the dimensions of the image
    rows, cols = image_array.shape
    
    # Compute the 2D Fourier transform of the image
    fft_image = fft2(image_array)
    fft_image_shifted = fftshift(fft_image)
    
    # Create the cross-shaped high-pass filter mask
    T = np.ones((rows, cols), dtype=float)
    
    # Define the central region which should be blocked
    center_row, center_col = rows // 2, cols // 2
    
    # Block a vertical band centered at the middle
    T[:, center_col - bandwidth:center_col + bandwidth] = 0
    
    # Block a horizontal band centered at the middle
    T[center_row - bandwidth:center_row + bandwidth, :] = 0
    
    # Apply the filter in the frequency domain
    filtered_fft_shifted = fft_image_shifted * T
    
    # Inverse shift and inverse FFT to transform back to spatial domain
    filtered_fft = ifftshift(filtered_fft_shifted)
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
