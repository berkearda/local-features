import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def draw_keypoints(img, keypoints, color=(0, 0, 255), thickness=2):
    if len(img.shape) == 2:
        img = img[:,:,None].repeat(3, 2)
    if keypoints is None:
        raise ValueError("Error! Keypoints should not be None")
    keypoints = np.array(keypoints)
    for p in keypoints.tolist():
        pos_x, pos_y = int(round(p[0])), int(round(p[1]))
        cv2.circle(img, (pos_x, pos_y), thickness, color, -1)
    return img

def draw_segments(img, segments, color=(255, 0, 0), thickness=2):
    for s in segments:
        p1 = (int(round(s[0])), int(round(s[1])))
        p2 = (int(round(s[2])), int(round(s[3])))
        cv2.line(img, p1, p2, color, thickness)
    return img

def plot_image_with_keypoints(fname_out, img, keypoints, color=(0, 0, 255), thickness=2):
    img = copy.deepcopy(img)
    img_keypoints = draw_keypoints(img, keypoints, color=color, thickness=thickness)
    cv2.imwrite(fname_out, img_keypoints)
    print("[LOG] Number of keypoints: {0}. Writing keypoints visualization to {1}".format(keypoints.shape[0], fname_out))

def plot_image_pair_with_matches(fname_out, img1, keypoints1, img2, keypoints2, matches):
    # construct full image
    import pdb
    assert img1.shape[0] == img2.shape[0]
    assert img1.shape[1] == img2.shape[1]
    h, w = img1.shape[0], img1.shape[1]
    img = np.concatenate([img1, img2], 1)
    img = img[:,:,None].repeat(3, 2)
    img = draw_keypoints(img, keypoints1, color=(0, 0, 255), thickness=2)
    img = draw_keypoints(img, keypoints2 + np.array([w, 0])[None,:], color=(0, 0, 255), thickness=2)
    segments = []
    segments.append(keypoints1[matches[:,0]])
    segments.append(keypoints2[matches[:,1]] + np.array([w, 0])[None,:])
    segments = np.concatenate(segments, axis=1)
    img = draw_segments(img, segments, color=(255, 0, 0), thickness=1)
    cv2.imwrite(fname_out, img)
    print("[LOG] Number of matches: {0}. Writing matches visualization to {1}".format(matches.shape[0], fname_out))

def visualize_gradients(img, Ix, Ix_blurred, row=None):
    """
    Function to visualize intensity and gradients (with and without blurring).

    Inputs:
    - img:        (h, w) grayscale image
    - Ix:         (h, w) gradient in x direction (before blurring)
    - Ix_blurred: (h, w) gradient in x direction (after blurring)
    - row:        specific row to visualize (default: middle row)
    """
    # Default row to visualize: middle row
    if row is None:
        row = img.shape[0] // 2

    # Professional style plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    blurred_intensity = ndimage.gaussian_filter(img, sigma=1.0)

    # 1. Original Intensity (without blur)
    axs[0, 0].plot(img[row, :], 'k')
    axs[0, 0].set_title('Intensity (Without blur)', fontsize=14)
    axs[0, 0].set_xlabel('Pixel position', fontsize=12)
    axs[0, 0].set_ylabel('Intensity', fontsize=12)

    # 2. Gradient (without blur)
    axs[0, 1].plot(Ix[row, :], 'k')
    axs[0, 1].set_title('Gradient (Without blur)', fontsize=14)
    axs[0, 1].set_xlabel('Pixel position', fontsize=12)
    axs[0, 1].set_ylabel('Gradient', fontsize=12)

    # 3. Blurred Intensity (optional, to match structure)
    axs[1, 0].plot(blurred_intensity[row, :], 'k')
    axs[1, 0].set_title('Intensity (With blur)', fontsize=14)
    axs[1, 0].set_xlabel('Pixel position', fontsize=12)
    axs[1, 0].set_ylabel('Intensity', fontsize=12)

    # 4. Blurred Gradient
    axs[1, 1].plot(Ix_blurred[row, :], 'k')
    axs[1, 1].set_title('Gradient (With blur)', fontsize=14)
    axs[1, 1].set_xlabel('Pixel position', fontsize=12)
    axs[1, 1].set_ylabel('Gradient', fontsize=12)

    # Professional layout adjustments
    plt.tight_layout()
    plt.show()


def plot_harris_response_with_values(C, step=20):
    """
    Plot the Harris response map with values printed for every Nth pixel.
    
    Parameters:
    - C: The Harris response matrix.
    - step: The interval at which to print the values (default is every 20th pixel).
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(C, cmap='gray')
    plt.title('Harris Response Map with Values')

    # Loop through the response matrix and print values at regular intervals
    h, w = C.shape
    for i in range(0, h, step):
        for j in range(0, w, step):
            plt.text(j, i, f'{C[i, j]:.2e}', color='red', fontsize=8, ha='center', va='center')

    plt.show()
