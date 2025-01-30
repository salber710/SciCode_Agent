from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift





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
    
    # Compute the 2D FFT of the image
    fft_image = fftshift(fft2(image_array))
    
    # Create a high-pass cross-shaped filter
    T = np.ones((rows, cols))
    
    # Define the center of the frequency transform
    center_row, center_col = rows // 2, cols // 2
    
    # Zero out the bandwidth rows and columns in the cross shape
    T[center_row - bandwidth:center_row + bandwidth, :] = 0
    T[:, center_col - bandwidth:center_col + bandwidth] = 0
    
    # Apply the filter to the FFT of the image
    filtered_fft_image = fft_image * T
    
    # Compute the inverse FFT to get the filtered image
    filtered_image = ifft2(ifftshift(filtered_fft_image))
    
    # Return the real part of the filtered image
    return T, np.real(filtered_image)


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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e