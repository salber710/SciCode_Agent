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
    
    # Create a frequency grid
    u = np.fft.fftfreq(rows, d=1/rows)
    v = np.fft.fftfreq(cols, d=1/cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # Shift the zero frequency component to the center
    U = fftshift(U)
    V = fftshift(V)
    
    # Create the cross-shaped high-pass filter
    T = np.ones((rows, cols), dtype=float)
    
    # Define the cross-shaped band region
    horizontal_band = (np.abs(U) < bandwidth)
    vertical_band = (np.abs(V) < bandwidth)
    
    # Apply the band to the filter
    T[horizontal_band] = 0
    T[vertical_band] = 0
    
    # Apply the Fourier transform to the image
    F_image = fft2(image_array)
    
    # Shift the zero frequency component to the center
    F_image_shifted = fftshift(F_image)
    
    # Apply the filter in the frequency domain
    F_filtered_shifted = F_image_shifted * T
    
    # Shift back the zero frequency component
    F_filtered = ifftshift(F_filtered_shifted)
    
    # Apply the inverse Fourier transform to get the filtered image
    filtered_image = np.real(ifft2(F_filtered))
    
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

except Exception as e:
    print(f'Error during execution: {str(e)}')
    raise e