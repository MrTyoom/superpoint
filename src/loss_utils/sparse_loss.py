import torch

from src.homography.apply import filter_points, warp_points
from src.loss_utils.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
from src.train_utils.correspondence_finder import create_non_correspondences
from src.train_utils.train_utils import crop_or_pad_choice, normPts


def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device="cpu"):
    if not uv:
        coor_cells = torch.stack((coor_cells[:, 1], coor_cells[:, 0]), dim=1)

    warped = warp_points(coor_cells, homographies, device)

    if not uv:
        warped = torch.stack((warped[:, :, 1], warped[:, :, 0]), dim=2)

    return warped


def get_coor_cells(Hc, Wc, device="cpu", uv=False):
    y, x = torch.meshgrid(
        torch.arange(Hc, device=device),
        torch.arange(Wc, device=device),
        indexing="ij",
    )

    coor = torch.stack((y, x), dim=-1).float().view(-1, 2)

    if uv:
        coor = coor[:, [1, 0]]

    return coor


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    uv_a_long = (
        torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
        torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1),
    )

    uv_b_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))
    return uv_a_long, uv_b_long


def uv_to_tuple(uv):
    return (uv[:, 0], uv[:, 1])


def tuple_to_1d(uv_tuple, W, uv=True):
    if uv:
        return uv_tuple[0] + uv_tuple[1] * W
    return uv_tuple[0] * W + uv_tuple[1]


def uv_to_1d_tensor(points, W, uv=True):
    if uv:
        return points[..., 0] + points[..., 1] * W
    return points[..., 0] * W + points[..., 1]


def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b, dist="cos", method="2d"):
    return PixelwiseContrastiveLoss.match_loss(
        image_a_pred, image_b_pred, matches_a, matches_b, dist=dist, method=method
    )[0]


def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10):
    uv_b_matches = uv_b_matches.squeeze()

    uv_b_matches_tuple = uv_to_tuple(uv_b_matches)
    uv_b_non_matches_tuple = create_non_correspondences(
        uv_b_matches_tuple,
        img_b_shape,
        num_non_matches_per_match=num_masked_non_matches_per_match,
        img_b_mask=None,
    )

    return create_non_matches(
        uv_to_tuple(uv_a),
        uv_b_non_matches_tuple,
        num_masked_non_matches_per_match,
    )


def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, dist="cos"):
    non_match_loss, num_hard_negatives, _, _ = PixelwiseContrastiveLoss.non_match_descriptor_loss(
        image_a_pred,
        image_b_pred,
        non_matches_a.long().squeeze(),
        non_matches_b.long().squeeze(),
        M=0.2,
        invert=True,
        dist=dist,
    )

    return non_match_loss.sum() / (num_hard_negatives + 1)


def descriptor_loss_sparse(
    descriptors,
    descriptors_warped,
    homographies,
    cell_size=8,
    lamda_d=250,
    num_matching_attempts=1000,
    num_masked_non_matches_per_match=10,
    dist="cos",
    method="2d",
    **config,
):
    device = homographies.device

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    H, W = Hc * cell_size, Wc * cell_size

    def reshape(x):
        return x.view(-1, Hc * Wc).transpose(0, 1).unsqueeze(0)

    image_a = reshape(descriptors)
    image_b = reshape(descriptors_warped)

    uv_a_cells = get_coor_cells(Hc, Wc, uv=True).to(device)
    uv_a_pixels = uv_a_cells * cell_size + cell_size / 2

    uv_b_pixels = warp_coor_cells_with_homographies(uv_a_pixels, homographies, uv=True, device=device).squeeze(0)

    uv_b_pixels = uv_b_pixels.round()
    uv_b_pixels, mask = filter_points(
        uv_b_pixels,
        torch.tensor([W, H], device=device),
        return_mask=True,
    )

    uv_a_cells = uv_a_cells[mask]

    uv_b_cells = (uv_b_pixels / cell_size).floor().long()

    choice = torch.tensor(
        crop_or_pad_choice(uv_b_cells.shape[0], num_matching_attempts, shuffle=True), dtype=torch.long, device=device
    )

    uv_a_cells = uv_a_cells[choice]
    uv_b_cells = uv_b_cells[choice]

    if method == "2d":
        matches_a = normPts(uv_a_cells.float(), torch.tensor([Wc, Hc], device=device).float())
        matches_b = normPts(uv_b_cells.float(), torch.tensor([Wc, Hc], device=device).float())
    else:
        matches_a = uv_to_1d_tensor(uv_a_cells, Wc)
        matches_b = uv_to_1d_tensor(uv_b_cells, Wc)

    match_loss = get_match_loss(
        descriptors,
        descriptors_warped,
        matches_a,
        matches_b,
        dist=dist,
        method=method,
    )

    uv_a_t, uv_b_t = get_non_matches_corr(
        (Hc, Wc),
        uv_a_cells.float(),
        uv_b_cells.float(),
        num_masked_non_matches_per_match,
    )

    non_a = tuple_to_1d(uv_a_t, Wc)
    non_b = tuple_to_1d(uv_b_t, Wc)

    non_match_loss = get_non_match_loss(
        image_a,
        image_b,
        non_a,
        non_b,
        dist=dist,
    )

    loss = lamda_d * match_loss + non_match_loss
    return loss, lamda_d * match_loss, non_match_loss


def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies, **options):
    loss, pos, neg = [], [], []

    for i in range(descriptors.shape[0]):
        lo, p, n = descriptor_loss_sparse(
            descriptors[i],
            descriptors_warped[i],
            homographies[i].float(),
            **options,
        )
        loss.append(lo)
        pos.append(p)
        neg.append(n)

    return torch.stack(loss).mean(), None, torch.stack(pos).mean(), torch.stack(neg).mean()
