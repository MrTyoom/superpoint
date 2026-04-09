import torch

from src.homography.apply import filter_points, homography_scaling_torch, warp_points
from src.loss_utils.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
from src.train_utils.correspondence_finder import create_non_correspondences
from src.train_utils.train_utils import crop_or_pad_choice, normPts


def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device="cpu"):
    warped_coor_cells = coor_cells
    if not uv:
        warped_coor_cells = torch.stack((warped_coor_cells[:, 1], warped_coor_cells[:, 0]), dim=1)  # (y, x) to (x, y)

    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if not uv:
        warped_coor_cells = torch.stack(
            (warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2
        )  # (batch, x, y) to (batch, y, x)

    return warped_coor_cells


def get_coor_cells(Hc, Wc, cell_size, device="cpu", uv=False):
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:, 1], coor_cells[:, 0]), dim=1)  # (y, x) to (x, y)

    return coor_cells.to(device)


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    """
    Simple wrapper for repeated code
    :param uv_a:
    :type uv_a:
    :param uv_b_non_matches:
    :type uv_b_non_matches:
    :param multiplier:
    :type multiplier:
    :return:
    :rtype:
    """
    uv_a_long = (
        torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
        torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1),
    )

    uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

    return uv_a_long, uv_b_non_matches_long


def uv_to_tuple(uv):
    return (uv[:, 0], uv[:, 1])


def tuple_to_1d(uv_tuple, W, uv=True):
    if uv:
        return uv_tuple[0] + uv_tuple[1] * W
    else:
        return uv_tuple[0] * W + uv_tuple[1]


def uv_to_1d(points, W, uv=True):
    if uv:
        return points[..., 0] + points[..., 1] * W
    else:
        return points[..., 0] * W + points[..., 1]


## calculate matches loss
def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b, dist="cos", method="1d"):
    match_loss, matches_a_descriptors, matches_b_descriptors = PixelwiseContrastiveLoss.match_loss(
        image_a_pred, image_b_pred, matches_a, matches_b, dist=dist, method=method
    )
    return match_loss


def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10, device="cpu"):
    ## sample non matches
    uv_b_matches = uv_b_matches.squeeze()
    uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
    uv_b_non_matches_tuple = create_non_correspondences(
        uv_b_matches_tuple, img_b_shape, num_non_matches_per_match=num_masked_non_matches_per_match, img_b_mask=None
    )

    uv_a_tuple, uv_b_non_matches_tuple = create_non_matches(
        uv_to_tuple(uv_a), uv_b_non_matches_tuple, num_masked_non_matches_per_match
    )
    return uv_a_tuple, uv_b_non_matches_tuple


def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, dist="cos"):
    ## non matches loss
    non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors = (
        PixelwiseContrastiveLoss.non_match_descriptor_loss(
            image_a_pred,
            image_b_pred,
            non_matches_a.long().squeeze(),
            non_matches_b.long().squeeze(),
            M=0.2,
            invert=True,
            dist=dist,
        )
    )
    non_match_loss = non_match_loss.sum() / (num_hard_negatives + 1)
    return non_match_loss


def descriptor_loss_sparse(
    descriptors,
    descriptors_warped,
    homographies,
    mask_valid=None,
    cell_size=8,
    device="cpu",
    descriptor_dist=4,
    lamda_d=250,
    num_matching_attempts=1000,
    num_masked_non_matches_per_match=10,
    dist="cos",
    method="1d",
    **config,
):
    """
    consider batches of descriptors
    :param descriptors:
        Output from descriptor head
        tensor [descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [descriptors, Hc, Wc]
    """

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    H, W = Hc * cell_size, Wc * cell_size
    img_shape = (H, W)

    # image_a_pred = descriptors.view(1, -1, Hc * Wc).transpose(1, 2)  # torch [batch_size, H*W, D]
    def descriptor_reshape(descriptors):
        descriptors = descriptors.view(-1, Hc * Wc).transpose(0, 1)  # torch [D, H, W] --> [H*W, d]
        descriptors = descriptors.unsqueeze(0)  # torch [1, H*W, D]
        return descriptors

    image_a_pred = descriptor_reshape(descriptors)  # torch [1, H*W, D]
    image_b_pred = descriptor_reshape(descriptors_warped)  # torch [batch_size, H*W, D]

    # matches
    uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True, device="cpu")

    homographies_H = homography_scaling_torch(homographies, img_shape, shift=(-1, -1))

    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to("cpu"), uv=True, device="cpu")

    uv_b_matches.round_()
    uv_b_matches = uv_b_matches.squeeze(0)

    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device="cpu"), return_mask=True)
    uv_a = uv_a[mask]

    shuffle = True
    if not shuffle:
        print("shuffle: ", shuffle)
    choice = crop_or_pad_choice(uv_b_matches.shape[0], num_matching_attempts, shuffle=shuffle)
    choice = torch.tensor(choice).type(torch.int64)
    uv_a = uv_a[choice]
    uv_b_matches = uv_b_matches[choice]

    if method == "2d":
        matches_a = normPts(uv_a, torch.tensor([Wc, Hc]).float())  # [u, v]
        matches_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float())
    else:
        matches_a = uv_to_1d(uv_a, Wc)
        matches_b = uv_to_1d(uv_b_matches, Wc)

    if method == "2d":
        match_loss = get_match_loss(
            descriptors, descriptors_warped, matches_a.to(device), matches_b.to(device), dist=dist, method="2d"
        )
    else:
        match_loss = get_match_loss(
            image_a_pred, image_b_pred, matches_a.long().to(device), matches_b.long().to(device), dist=dist
        )

    # get non matches correspondence
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(
        img_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=num_masked_non_matches_per_match
    )

    non_matches_a = tuple_to_1d(uv_a_tuple, Wc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Wc)

    non_match_loss = get_non_match_loss(
        image_a_pred, image_b_pred, non_matches_a.to(device), non_matches_b.to(device), dist=dist
    )

    loss = lamda_d * match_loss + non_match_loss
    return loss, lamda_d * match_loss, non_match_loss


def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies, **options):
    loss = []
    pos_loss = []
    neg_loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        losses = descriptor_loss_sparse(
            descriptors[i],
            descriptors_warped[i],
            # torch.tensor(homographies[i], dtype=torch.float32), **options)
            homographies[i].type(torch.float32),
            **options,
        )
        loss.append(losses[0])
        pos_loss.append(losses[1])
        neg_loss.append(losses[2])
    loss, pos_loss, neg_loss = torch.stack(loss), torch.stack(pos_loss), torch.stack(neg_loss)
    return loss.mean(), None, pos_loss.mean(), neg_loss.mean()
