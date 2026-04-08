import torch


dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


def pytorch_rand_select_pixel(width, height, num_samples=1):
    two_rand_numbers = torch.rand(2, num_samples)
    two_rand_numbers[0, :] = two_rand_numbers[0, :] * width
    two_rand_numbers[1, :] = two_rand_numbers[1, :] * height
    two_rand_ints = torch.floor(two_rand_numbers).type(dtype_long)
    return (two_rand_ints[0], two_rand_ints[1])


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
    return (cond * x_1) + ((1 - cond) * x_2)


def create_non_correspondences(uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another image,
    and generates non-matches by just sampling in image space.

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
        - The first element of the tuple is all "u" pixel positions, and the right one is all "v" positions
        - Each torch.FloatTensor is of shape torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds to the row for the match in uv_a
    """
    image_width = img_b_shape[1]
    image_height = img_b_shape[0]
    # print("uv_b_matches: ", uv_b_matches)
    if uv_b_matches is None:
        return None

    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return pytorch_rand_select_pixel(
            width=image_width, height=image_height, num_samples=num_matches * num_non_matches_per_match
        )

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.view(-1, 1).squeeze(1)
        mask_b_indices_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indices_flat) == 0:
            print("warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches * num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples) * len(mask_b_indices_flat)
            rand_indices_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indices_flat = torch.index_select(mask_b_indices_flat, 0, rand_indices_b).squeeze(1)
            uv_b_non_matches = (
                randomized_mask_b_indices_flat % image_width,
                randomized_mask_b_indices_flat / image_width,
            )
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()

    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = (
        uv_b_non_matches[0].view(num_matches, num_non_matches_per_match),
        uv_b_non_matches[1].view(num_matches, num_non_matches_per_match),
    )

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(uv_b_matches[0].repeat(num_non_matches_per_match, 1))
    copied_uv_b_matches_1 = torch.t(uv_b_matches[1].repeat(num_non_matches_per_match, 1))

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.contiguous().view(-1, 1)
    diffs_1_flattened = diffs_1.contiguous().view(-1, 1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)

    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened) * num_pixels_too_close

    # determine which pixels are too close to being matches
    need_to_be_perturbed = where(diffs_0_flattened < threshold, ones, need_to_be_perturbed)
    need_to_be_perturbed = where(diffs_1_flattened < threshold, ones, need_to_be_perturbed)

    minimal_perturb = num_pixels_too_close / 2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed)) * 2).floor() * (
        minimal_perturb * 2
    ) - minimal_perturb
    std_dev = 10
    random_vector = torch.randn(len(need_to_be_perturbed)) * std_dev + minimal_perturb_vector
    perturb_vector = need_to_be_perturbed * random_vector

    uv_b_non_matches_0_flat = uv_b_non_matches[0].view(-1, 1).type(dtype_float).squeeze(1)
    uv_b_non_matches_1_flat = uv_b_non_matches[1].view(-1, 1).type(dtype_float).squeeze(1)

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = image_width * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = where(
        uv_b_non_matches_0_flat > upper_bound_vec, uv_b_non_matches_0_flat - upper_bound_vec, uv_b_non_matches_0_flat
    )

    uv_b_non_matches_0_flat = where(
        uv_b_non_matches_0_flat < lower_bound_vec, uv_b_non_matches_0_flat + upper_bound_vec, uv_b_non_matches_0_flat
    )

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = image_height * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = where(
        uv_b_non_matches_1_flat > upper_bound_vec, uv_b_non_matches_1_flat - upper_bound_vec, uv_b_non_matches_1_flat
    )

    uv_b_non_matches_1_flat = where(
        uv_b_non_matches_1_flat < lower_bound_vec, uv_b_non_matches_1_flat + upper_bound_vec, uv_b_non_matches_1_flat
    )

    return (
        uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
        uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match),
    )
