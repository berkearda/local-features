import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    diff = desc1[:, None, :] - desc2[None, :, :]
    distances = np.sum(diff ** 2, axis=-1)  # Sum over the feature dimension
    return distances

def match_descriptors(desc1, desc2, method = "ratio", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        indices = np.argmin(distances, axis=1) 
        matches = np.column_stack((np.arange(len(indices)), indices))
        
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        indices_1to2 = np.argmin(distances, axis=1)
        indices_2to1 = np.argmin(distances, axis=0)
        mutual_mask = np.arange(len(indices_1to2)) == indices_2to1[indices_1to2]
        matches = np.column_stack((np.arange(len(indices_1to2))[mutual_mask], indices_1to2[mutual_mask]))

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        smallest_values = np.min(distances, axis=1)
        smallest_second_values = np.partition(distances, 2, axis=1)[:, 1]
        ratio = smallest_values / smallest_second_values
        mask = ratio < ratio_thresh
        matches = np.column_stack((np.arange(len(mask))[mask], np.argmin(distances, axis=1)[mask]))
    else:
        raise NotImplementedError
    return matches

