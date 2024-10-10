import numpy as np
from matplotlib import pyplot as plt
from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
from functions.vis_utils import visualize_gradients, plot_harris_response_with_values  #for the visualize_gradients function

def apply_border_mask(C, border_size=20):
    """
    Apply a border mask to ignore corner responses near the image edges.
    
    Parameters:
    - C: Harris response matrix
    - border_size: the width of the border to mask out (in pixels)
    
    Returns:
    - masked_C: Harris response matrix with the borders zeroed out
    """
    masked_C = C.copy()
    masked_C[:border_size, :] = 0  
    masked_C[-border_size:, :] = 0  
    masked_C[:, :border_size] = 0  
    masked_C[:, -border_size:] = 0  
    return masked_C


def create_gaussian_kernel(size, sigma):
    """
    Create a 2D Gaussian kernel.
    - size: Kernel size (should be an odd number)
    - sigma: Standard deviation of the Gaussian distribution
    Returns:
    - A 2D Gaussian kernel
    """
    ax = np.arange(-(size // 2), (size // 2) + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel) 

# Harris corner detector
def extract_harris(img, sigma =1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.

    central_diff_x = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2
    central_diff_y = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2

    Ix = signal.convolve2d(img, central_diff_x, mode='same')
    Iy = signal.convolve2d(img, central_diff_y, mode='same')

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy

    # 2. (Optional) Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    Ixx_blurred = ndimage.gaussian_filter(Ixx, sigma=sigma)
    Iyy_blurred = ndimage.gaussian_filter(Iyy, sigma=sigma)
    Ixy_blurred = ndimage.gaussian_filter(Ixy, sigma=sigma)

    visualize_gradients(img, Ix, Ixx_blurred)

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here
    # You may refer to cv2.GaussianBlur or scipy.signal.convolve2d to perform the weighted sum
    # Apply the filter over the components as a whole (simulating weighted sum over W)
    # Create Gaussian kernel for weighted sum
    kernel_size = int(4 * sigma + 1)
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)

    # Perform the weighted sum using convolve2d
    Ixx_weighted = signal.convolve2d(Ixx_blurred, gaussian_kernel, mode='same')
    Iyy_weighted = signal.convolve2d(Iyy_blurred, gaussian_kernel, mode='same')
    Ixy_weighted = signal.convolve2d(Ixy_blurred, gaussian_kernel, mode='same')
    
    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here
    det_M = (Ixx_weighted * Iyy_weighted) - (Ixy_weighted ** 2)
    trace_M = Ixx_weighted + Iyy_weighted
    C = det_M - k * (trace_M ** 2)
    C = apply_border_mask(C, border_size=7)
    # plot_harris_response_with_values(C, step=100)
    plt.imshow(C, cmap='gray')
    plt.title('Harris Response Map')
    plt.show()

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    C_max = ndimage.maximum_filter(C, size=3)
    corners = (C == C_max) & (C > thresh)
    y, x = np.where(corners)
    corners = np.stack((x, y), axis=-1)
    return corners, C

