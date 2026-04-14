import torch


def get_center_crop_bounds(height, width, crop_h, crop_w):
    crop_h = min(crop_h, height)
    crop_w = min(crop_w, width)

    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    bottom = top + crop_h
    right = left + crop_w

    return left, right, top, bottom


def crop_tensor(tens, bounds):
    left, right, top, bottom = bounds
    return tens[:, top:bottom, left:right]


def crop_points(points, bounds, crop_h, crop_w):
    left, _, top, _ = bounds

    pts = points.clone()
    pts[:, 0] -= left
    pts[:, 1] -= top

    low_bound = (pts[:, 0] >= 0) & (pts[:, 1] >= 0)
    up_bound = (pts[:, 0] < crop_w) & (pts[:, 1] < crop_h)
    valid = low_bound & up_bound

    return pts[valid]


def crop_data(images, warped_img, mask, mask_w, pts, warped_pts, bounds, crop_h, crop_w):  # noqa: WPS211
    images = crop_tensor(images, bounds)
    warped_img = crop_tensor(warped_img, bounds)

    mask = crop_tensor(mask, bounds)
    mask_w = crop_tensor(mask_w, bounds)

    pts = crop_points(pts, bounds, crop_h, crop_w)
    warped_pts = crop_points(warped_pts, bounds, crop_h, crop_w)

    return (images, warped_img), (mask, mask_w), pts, warped_pts


def crop_homography(homography, left, top):
    homo = torch.tensor(
        [[1, 0, -left], [0, 1, -top], [0, 0, 1]],
        dtype=torch.float32,
    )
    homo = homo.unsqueeze(0)

    homo_inv = torch.tensor([[1, 0, left], [0, 1, top], [0, 0, 1]], dtype=torch.float32)  # noqa: WPS221
    homo_inv = homo_inv.unsqueeze(0)

    homo_crop = homo @ homography @ homo_inv
    inv_homo_crop = torch.linalg.inv(homo_crop)

    return homo_crop, inv_homo_crop
