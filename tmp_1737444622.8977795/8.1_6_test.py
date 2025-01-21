import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift



# Background: Fourier Optics involves transforming images into the frequency domain using the Fourier transform, 
# allowing us to manipulate specific frequency components of the image. A spatial filter in the frequency domain 
# can be used to selectively block or pass specific frequency components. In this case, a cross-shaped high-pass filter 
# is designed to allow high frequencies to pass while blocking lower frequencies, except along two orthogonal lines 
# (the cross shape). High-pass filters are used to remove low-frequency noise and enhance edges or fine details in images. 
# The cross shape is designed to preserve certain directional information while filtering out low-frequency background noise. 
# The Fourier transform of an image is shifted to center the zero-frequency component for easier filter design. After 
# applying the filter, the inverse Fourier transform is used to convert the image back to the spatial domain.



def apply_cshband_pass_filter(image_array, bandwidth):
    '''Applies a cross shaped high band pass filter to the given image array based on the frequency bandwidth.
    Input:
    image_array: float;2D numpy array, the input image.
    bandwitdh: bandwidth of cross-shaped filter, int
    Ouput:
    T: 2D numpy array of float, The spatial filter used.
    filtered_image: 2D numpy array of float, the filtered image in the original domain.
    '''
    
    # Get the dimensions of the image
    rows, cols = image_array.shape
    
    # Perform Fourier transform and shift zero frequency component to the center
    f_transform = fftshift(fft2(image_array))
    
    # Initialize the filter mask with ones (allowing all frequencies by default)
    T = np.ones((rows, cols), dtype=float)
    
    # Get the center of the frequency domain
    center_row, center_col = rows // 2, cols // 2
    
    # Apply the cross-shaped high band pass filter by setting low frequency components to zero
    # Vertical line
    T[center_row-bandwidth:center_row+bandwidth+1, :] = 0
    # Horizontal line
    T[:, center_col-bandwidth:center_col+bandwidth+1] = 0
    
    # Ensure the bandwidth frequency itself is not included in the mask
    T[center_row-bandwidth:center_row+bandwidth+1, center_col-bandwidth:center_col+bandwidth+1] = 1
    
    # Apply the filter mask
    filtered_transform = T * f_transform
    
    # Perform inverse Fourier transform and shift
    filtered_image = np.abs(ifft2(ifftshift(filtered_transform)))
    
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
