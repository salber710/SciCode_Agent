from scicode.parse.parse import process_hdf5_to_tuple
import numpy as np

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift





def apply_cshband_pass_filter(image_array, bandwidth):
    rows, cols = image_array.shape
    u = np.fft.fftfreq(rows)
    v = np.fft.fftfreq(cols)
    U, V = np.meshgrid(u, v, indexing='ij')
    U = fftshift(U)
    V = fftshift(V)

    # Create a cross-shaped high-pass filter with a specified bandwidth
    T = np.ones((rows, cols), dtype=float)
    # Set the cross-shaped region to zero, excluding the exact bandwidth frequency
    T[np.abs(U) <= bandwidth] = 0
    T[np.abs(V) <= bandwidth] = 0
    T[np.abs(U) == bandwidth] = 1
    T[np.abs(V) == bandwidth] = 1

    # Fourier transform of the image
    F_image = fft2(image_array)
    F_image_shifted = fftshift(F_image)

    # Apply the filter in the frequency domain
    F_filtered_shifted = F_image_shifted * T

    # Inverse Fourier transform to get the filtered image
    F_filtered = ifftshift(F_filtered_shifted)
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