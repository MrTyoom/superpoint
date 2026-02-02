# torch
import torch

# math
import numpy as numpy
import numpy as np
import math
from numpy.linalg import inv
import random

# io
from PIL import Image

# torchvision
import sys
# sys.path.insert(0, '../pytorch-segmentation-detection/vision/') # from subrepo
from torchvision import transforms
from utils.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
# from dense_correspondence_manipulation.utils.constants import *

# turns out to be faster to do this match generation on the CPU
# for the general size of params we expect
# also this will help by not takinfg up GPU memory, 
# allowing batch sizes to stay large
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

def pytorch_rand_select_pixel(width,height,num_samples=1):
    two_rand_numbers = torch.rand(2,num_samples)
    two_rand_numbers[0,:] = two_rand_numbers[0,:]*width
    two_rand_numbers[1,:] = two_rand_numbers[1,:]*height
    two_rand_ints    = torch.floor(two_rand_numbers).type(dtype_long)
    return (two_rand_ints[0], two_rand_ints[1])

def get_default_K_matrix():
    K = numpy.zeros((3,3))
    K[0,0] = 533.6422696034836 # focal x
    K[1,1] = 534.7824445233571 # focal y
    K[0,2] = 319.4091030774892 # principal point x
    K[1,2] = 236.4374299691866 # principal point y
    K[2,2] = 1.0
    return K

def get_body_to_rdf():
    body_to_rdf = numpy.zeros((3,3))
    body_to_rdf[0,1] = -1.0
    body_to_rdf[1,2] = -1.0
    body_to_rdf[2,0] = 1.0
    return body_to_rdf

def invert_transform(transform4):
    transform4_copy = numpy.copy(transform4)
    R = transform4_copy[0:3,0:3]
    R = numpy.transpose(R)
    transform4_copy[0:3,0:3] = R
    t = transform4_copy[0:3,3]
    inv_t = -1.0 * numpy.transpose(R).dot(t)
    transform4_copy[0:3,3] = inv_t
    return transform4_copy

def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0,:]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3,ones_row),0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]

def random_sample_from_masked_image(img_mask, num_samples):
    """
    Samples num_samples (row, column) convention pixel locations from the masked image
    Note this is not in (u,v) format, but in same format as img_mask
    :param img_mask: numpy.ndarray
        - masked image, we will select from the non-zero entries
        - shape is H x W
    :param num_samples: int
        - number of random indices to return
    :return: List of np.array
    """
    idx_tuple = img_mask.nonzero()
    num_nonzero = len(idx_tuple[0])
    if num_nonzero == 0:
        empty_list = []
        return empty_list
    rand_inds = random.sample(range(0,num_nonzero), num_samples)

    sampled_idx_list = []
    for i, idx in enumerate(idx_tuple):
        sampled_idx_list.append(idx[rand_inds])

    return sampled_idx_list

def random_sample_from_masked_image_torch(img_mask, num_samples):
    """

    :param img_mask: Numpy array [H,W] or torch.Tensor with shape [H,W]
    :type img_mask:
    :param num_samples: an integer
    :type num_samples:
    :return: tuple of torch.LongTensor in (u,v) format. Each torch.LongTensor has shape
    [num_samples]
    :rtype:
    """

    image_height, image_width = img_mask.shape

    if isinstance(img_mask, np.ndarray):
        img_mask_torch = torch.from_numpy(img_mask).float()
    else:
        img_mask_torch = img_mask

    # This code would randomly subsample from the mask
    mask = img_mask_torch.view(image_width*image_height,1).squeeze(1)
    mask_indices_flat = torch.nonzero(mask)
    if len(mask_indices_flat) == 0:
        return (None, None)

    rand_numbers = torch.rand(num_samples)*len(mask_indices_flat)
    rand_indices = torch.floor(rand_numbers).long()
    uv_vec_flattened = torch.index_select(mask_indices_flat, 0, rand_indices).squeeze(1)
    uv_vec = PixelwiseContrastiveLoss.flattened_pixel_locations_to_u_v_static(uv_vec_flattened, image_width)
    return uv_vec

def pinhole_projection_image_to_world(uv, z, K):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :return: (x,y,z) in camera frame
    :rtype: numpy.array size (3,)
    """

    u_v_1 = np.array([uv[0], uv[1], 1])
    pos = z * np.matmul(inv(K),u_v_1)
    return pos

def pinhole_projection_world_to_image(world_pos, K, camera_to_world=None):
    """
    Projects from world position to camera coordinates
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    :param world_pos:
    :type world_pos:
    :param K:
    :type K:
    :return:
    :rtype:
    """

    world_pos_vec = np.append(world_pos, 1)

    # transform to camera frame if camera_to_world is not None
    if camera_to_world is not None:
        world_pos_vec = np.dot(np.linalg.inv(camera_to_world), world_pos_vec)

    # scaled position is [X/Z, Y/Z, 1] where X,Y,Z is the position in camera frame
    scaled_pos = np.array([world_pos_vec[0]/world_pos_vec[2], world_pos_vec[1]/world_pos_vec[2], 1])
    uv = np.dot(K, scaled_pos)[:2]
    return uv



# in torch 0.3 we don't yet have torch.where(), although this
# is there in 0.4 (not yet stable release)
# for more see: https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
def where(cond, x_1, x_2):
    """
    We follow the torch.where implemented in 0.4.
    See http://pytorch.org/docs/master/torch.html?highlight=where#torch.where

    For more discussion see https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8


    Return a tensor of elements selected from either x_1 or x_2, depending on condition.
    :param cond: cond should be tensor with entries [0,1]
    :type cond:
    :param x_1: torch.Tensor
    :type x_1:
    :param x_2: torch.Tensor
    :type x_2:
    :return:
    :rtype:
    """
    cond = cond.type(dtype_float)    
    return (cond * x_1) + ((1-cond) * x_2)

def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image, and generates non-matches by just sampling in image space.

    Optionally, the non-matches can be sampled from a mask for image b.

    Returns non-matches as pixel positions in image b.

    Please see 'coordinate_conventions.md' documentation for an explanation of pixel coordinate conventions.

    ## Note that arg uv_b_matches are the outputs of batch_find_pixel_correspondences()

    :param uv_b_matches: tuple of torch.FloatTensors, where each FloatTensor is length n, i.e.:
        (torch.FloatTensor, torch.FloatTensor)

    :param img_b_shape: tuple of (H,W) which is the shape of the image

    (optional)
    :param num_non_matches_per_match: int

    (optional)
    :param img_b_mask: torch.FloatTensor (can be cuda or not)
        - masked image, we will select from the non-zero entries
        - shape is H x W
     
    :return: tuple of torch.FloatTensors, i.e. (torch.FloatTensor, torch.FloatTensor).
        - The first element of the tuple is all "u" pixel positions, and the right element of the tuple is all "v" positions
        - Each torch.FloatTensor is of shape torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds to the row for the match in uv_a
    """
    image_width  = img_b_shape[1]
    image_height = img_b_shape[0]
    # print("uv_b_matches: ", uv_b_matches)
    if uv_b_matches == None:
        return None

    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return pytorch_rand_select_pixel(width=image_width,height=image_height, 
            num_samples=num_matches*num_non_matches_per_match)

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.view(-1,1).squeeze(1)
        mask_b_indices_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indices_flat) == 0:
            print("warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches*num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples)*len(mask_b_indices_flat)
            rand_indices_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indices_flat = torch.index_select(mask_b_indices_flat, 0, rand_indices_b).squeeze(1)
            uv_b_non_matches = (randomized_mask_b_indices_flat%image_width, randomized_mask_b_indices_flat/image_width)
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()
    
    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = (uv_b_non_matches[0].view(num_matches,num_non_matches_per_match), uv_b_non_matches[1].view(num_matches,num_non_matches_per_match))

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.contiguous().view(-1,1)
    diffs_1_flattened = diffs_1.contiguous().view(-1,1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)


    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened)*num_pixels_too_close

    # determine which pixels are too close to being matches
    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb        = num_pixels_too_close/2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed))*2).floor()*(minimal_perturb*2)-minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed))*std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed*random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1,1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1,1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = image_width*1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat > upper_bound_vec, 
        uv_b_non_matches_0_flat - upper_bound_vec, 
        uv_b_non_matches_0_flat)

    uv_b_non_matches_0_flat = where(uv_b_non_matches_0_flat < lower_bound_vec, 
        uv_b_non_matches_0_flat + upper_bound_vec, 
        uv_b_non_matches_0_flat)

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = image_height*1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat > upper_bound_vec, 
        uv_b_non_matches_1_flat - upper_bound_vec, 
        uv_b_non_matches_1_flat)

    uv_b_non_matches_1_flat = where(uv_b_non_matches_1_flat < lower_bound_vec, 
        uv_b_non_matches_1_flat + upper_bound_vec, 
        uv_b_non_matches_1_flat)

    return (uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
        uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match))
