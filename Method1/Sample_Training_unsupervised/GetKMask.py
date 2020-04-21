import sys
import numpy as np


##################################################
def createkSpaceMask (im_size, num_half_lines):
    """ Compute mask of radial lines for given image size
    :param im_size: image size
    :param num_half_lines: number of half radial lines
    :return: sampling mask
    """
    
    if im_size[0] < 32 or im_size[1] < 32:
        sys.exit("side < 32")

    if num_half_lines < 20:
        sys.exit("numOfLines < 10")
        
    center = np.floor((im_size + 2) / 2)
    half_diagonal = np.linalg.norm(im_size) / 2
    step_length = 0.5
    num_steps = int(np.round(half_diagonal / step_length + 1))
    sampling_mask = np.zeros(im_size, float)

    for lineNum in range(num_half_lines):
        theta = 2 * np.pi * lineNum / num_half_lines
        direction = np.array([np.cos(theta), np.sin(theta)])
        for stepNum in range(num_steps):
            location = np.round(center + direction * stepNum * step_length).astype(int)
            if (location[0] >= 0) and (location[0] < im_size[0]) and (location[1] >= 0) and (location[1] < im_size[1]):
                sampling_mask[location[0], location[1]] = 1

    # take the center of kspace to the corners
    sampling_mask = np.fft.fftshift(sampling_mask)

    return sampling_mask

