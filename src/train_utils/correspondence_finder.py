import torch


def pytorch_rand_select_pixel(width, height, num_samples=1, device="cpu"):
    rand = torch.rand(2, num_samples, device=device)
    rand[0] *= width
    rand[1] *= height
    rand = torch.floor(rand).long()
    return rand[0], rand[1]


def where(cond, x1, x2):
    return torch.where(cond.bool(), x1, x2)


def create_non_correspondences(
    uv_b_matches,
    img_b_shape,
    num_non_matches_per_match=100,
    img_b_mask=None,
):
    if uv_b_matches is None:
        return None

    device = uv_b_matches[0].device
    image_height, image_width = img_b_shape
    num_matches = len(uv_b_matches[0])

    def get_random():
        return pytorch_rand_select_pixel(
            image_width,
            image_height,
            num_matches * num_non_matches_per_match,
            device=device,
        )

    if img_b_mask is not None:
        img_b_mask = img_b_mask.to(device)
        flat = img_b_mask.reshape(-1)
        valid_ids = torch.nonzero(flat, as_tuple=False).squeeze(1)

        if len(valid_ids) == 0:
            uv_b_non_matches = get_random()
        else:
            total = num_matches * num_non_matches_per_match
            rand_ids = torch.randint(0, len(valid_ids), (total,), device=device)
            sampled = valid_ids[rand_ids]

            uv_b_non_matches = (
                sampled % image_width,
                sampled // image_width,
            )
    else:
        uv_b_non_matches = get_random()

    u = uv_b_non_matches[0].view(num_matches, num_non_matches_per_match).float()
    v = uv_b_non_matches[1].view(num_matches, num_non_matches_per_match).float()

    match_u = uv_b_matches[0].repeat(num_non_matches_per_match, 1).T
    match_v = uv_b_matches[1].repeat(num_non_matches_per_match, 1).T

    diff_u = (match_u - u).abs().reshape(-1)
    diff_v = (match_v - v).abs().reshape(-1)

    threshold = 1.0
    need = (diff_u < threshold) | (diff_v < threshold)

    minimal = threshold / 2
    sign = torch.randint(0, 2, (len(need),), device=device).float()
    sign = sign * (2 * minimal) - minimal

    noise = torch.randn(len(need), device=device) * 10 + sign
    perturb = need.float() * noise

    u = u.reshape(-1) + perturb
    v = v.reshape(-1) + perturb

    u = torch.remainder(u, image_width - 1)
    v = torch.remainder(v, image_height - 1)

    return (
        u.view(num_matches, num_non_matches_per_match),
        v.view(num_matches, num_non_matches_per_match),
    )
