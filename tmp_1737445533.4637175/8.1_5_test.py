try:
    import numpy as np
    from numpy.fft import fft2, ifft2, fftshift, ifftshift
    
    
    
    # Background: Spatial filtering is a technique used in image processing to enhance or suppress
    # certain features of an image. In the frequency domain, this involves modifying the Fourier
    # transform of the image. A high pass filter allows high-frequency components to pass through
    # while attenuating low-frequency components. A cross-shaped band high-pass filter specifically
    # targets frequencies along the axes, creating a cross-like pattern in the frequency domain.
    # This can be achieved using Fourier optics by applying a mask in the frequency domain and then
    # transforming back to the spatial domain. The filter mask should exclude the frequencies within
    # the defined bandwidth, effectively creating a notch filter along the cross axes.
    
    
    
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
    
        # Create the cross-shaped high-pass filter mask
        T = np.ones((rows, cols))
    
        # Calculate the center of the frequency domain
        center_row, center_col = rows // 2, cols // 2
    
        # Apply the bandwidth to create the cross shape filter
        # Zero out the cross along the horizontal and vertical axes
        T[center_row - bandwidth:center_row + bandwidth + 1, :] = 0
        T[:, center_col - bandwidth:center_col + bandwidth + 1] = 0
    
        # Transform the image to the frequency domain
        F_image = fft2(image_array)
        F_image_shifted = fftshift(F_image)
    
        # Apply the cross-shaped filter mask
        filtered_F_image_shifted = F_image_shifted * T
    
        # Transform back to the spatial domain
        filtered_F_image = ifftshift(filtered_F_image_shifted)
        filtered_image = ifft2(filtered_F_image).real
    
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
