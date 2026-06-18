import torch
from omegaconf import DictConfig

from src.homography.homography_utils import crop_homography
from src.types import Tensor, TupleInt


def get_perspective_transform(points_src: Tensor, points_dst: Tensor) -> Tensor:
    dtype = points_src.dtype
    device = points_src.device
    # create the lhs tensor with shape # Bx8x8
    batch_size = points_src.shape[0]  # batch_size
    matrix = torch.empty(batch_size, 8, 8, device=device, dtype=dtype)
    # we need to perform in batch
    zeros = torch.zeros(batch_size, device=device, dtype=dtype)
    ones = torch.ones(batch_size, device=device, dtype=dtype)

    for it in range(4):
        x1, y1 = points_src[..., it, 0], points_src[..., it, 1]  # Bx4
        x2, y2 = points_dst[..., it, 0], points_dst[..., it, 1]  # Bx4

        vec = [x1, y1, ones, zeros, zeros, zeros, -x1 * x2, -y1 * x2]
        matrix[:, 2 * it] = torch.stack(vec, -1)
        vec = [zeros, zeros, zeros, x1, y1, ones, -x1 * y2, -y1 * y2]
        matrix[:, 2 * it + 1] = torch.stack(vec, -1)

    # the rhs tensor
    vector = points_dst.reshape(-1, 8, 1)
    # solve the system Ax = b
    solution = torch.linalg.solve(matrix, vector).to(dtype)

    # create variable to return the Bx3x3 transform
    perspective = torch.empty(batch_size, 9, device=device, dtype=dtype)
    perspective[..., :8] = solution[..., 0]  # Bx8
    perspective[..., -1].fill_(1)

    return perspective.view(-1, 3, 3)  # Bx3x3


def normal_transform_pixel(height: int, width: int, device: torch.device, eps: float = 1e-14) -> Tensor:
    vec = [[1, 0, -1], [0, 1, -1], [0, 0, 1]]
    tr_mat = torch.tensor(vec, device=device, dtype=torch.float)

    # prevent divide by zero bugs
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2 / width_denom  # noqa: WPS432
    tr_mat[1, 1] = tr_mat[1, 1] * 2 / height_denom  # noqa: WPS432

    return tr_mat.unsqueeze(0)  # 1x3x3


def normalize_homography(homography: Tensor, dsize_src: TupleInt, dsize_dst: TupleInt) -> Tensor:
    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    device = homography.device

    # compute the transformation pixel/norm for src/dst
    src_pix = normal_transform_pixel(src_h, src_w, device)
    dst_pix = normal_transform_pixel(dst_h, dst_w, device)

    # compute chain transformations
    norm_homography = dst_pix @ (homography @ torch.inverse(src_pix))

    return norm_homography


def get_rotations(min_angle: float, max_angle: float, batch_size: int) -> Tensor:
    # случайные углы из диапазона [min_angle, max_angle) в радианах
    deg_angles = (max_angle - min_angle) * torch.rand(batch_size) + min_angle
    rad_angles = (torch.pi / 180.0) * deg_angles  # noqa: WPS432

    # матрицы вращения
    rots = []

    for angle in rad_angles:
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        rots.append(torch.tensor([[cos, -sin], [sin, cos]]))

    rotation_matrices = torch.stack(rots)

    return rotation_matrices


def affine_transform_points(
    src_pts: Tensor, rotation_matrices: Tensor, image_size: tuple[int, int], min_shift: float, max_shift: float
) -> Tensor:
    width, height = image_size

    # поворот точек вокруг центра изображений
    center = torch.tensor([width / 2, height / 2], dtype=torch.float)
    rot_pts = torch.einsum("bij,bkj->bki", rotation_matrices, src_pts - center) + center

    # максимальный и минимальный сдвиг точек в аффинном преобразовании в пикселях
    min_shift = width * min_shift
    max_shift = width * max_shift

    # добавим случайный сдвиг точек для аффинного преобразования
    shift = (max_shift - min_shift) * torch.rand(*rot_pts.shape) + min_shift
    dst_pts = rot_pts + shift

    return dst_pts


def calc_homography_grid(src_pts: Tensor, dst_pts: Tensor, batch: Tensor) -> tuple[Tensor, ...]:
    batch_size, channels, height, width = batch.shape
    device = batch.device

    # вычисление матриц гомографий и обратных к ним
    homography = get_perspective_transform(src_pts, dst_pts)
    homography_pixel = homography.clone()

    homography = normalize_homography(homography, (height, width), (height, width))
    inv_homography = torch.inverse(homography)

    homography = homography.to(device)
    inv_homography = inv_homography.to(device)
    homography_pixel = homography_pixel.to(device)

    # создание интерполяционных сеток для геометрических преобразований изображений
    grid = torch.affine_grid_generator(homography[:, :2, :], [batch_size, channels, height, width], align_corners=True)
    inv_grid = torch.affine_grid_generator(
        inv_homography[:, :2, :], [batch_size, channels, height, width], align_corners=True
    )

    return grid, inv_grid, homography, inv_homography, homography_pixel


def adjust_brightness(batch: Tensor, min_brightness: int, max_brightness: int, batch_size: int) -> Tensor:
    rnd_vec = torch.rand(batch_size, device=batch.device)
    vec: Tensor = (max_brightness - min_brightness) * rnd_vec + min_brightness
    batch = (vec.view(batch_size, 1, 1, 1) * batch).clamp(0, 1.0)

    return batch


def adjust_contrast(batch: Tensor, min_contrast: float, max_contrast: float, batch_size: int) -> Tensor:
    rnd_vec = torch.rand(batch_size, device=batch.device)
    vec: Tensor = (max_contrast - min_contrast) * rnd_vec + min_contrast
    vec = vec.view(batch_size, 1, 1, 1)
    mean = torch.mean(batch, dim=(2, 3), keepdim=True)
    batch = (vec * batch + (1.0 - vec) * mean).clamp(0, 1.0)

    return batch


def center_crop(batch: Tensor, crop_size: TupleInt) -> Tensor:
    height, width = batch.shape[-2:]
    crop_h, crop_w = crop_size

    start_h = (height - crop_h) // 2
    start_w = (width - crop_w) // 2

    crop = batch[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w]

    return crop, (start_h, start_w)


class Augmentation:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.grid = None
        self.inv_grid = None

        self.homography = None
        self.homography_pixel = None
        self.inv_homography = None

    def __call__(self, batch: Tensor) -> Tensor:
        # геометрическая аугментация
        self.calculate_transform(batch)

        batch = torch.grid_sampler(batch, self.inv_grid, 0, 0, align_corners=True)
        batch_size = batch.shape[0]

        # аугментация яркости
        batch = adjust_brightness(batch, self.cfg.min_brightness, self.cfg.max_brightness, batch_size)
        # аугментация контрастности
        batch = adjust_contrast(batch, self.cfg.min_contrast, self.cfg.max_contrast, batch_size)

        return batch

    def calculate_transform(self, batch: Tensor):
        batch_size, channels, height, width = batch.shape

        # матрицы вращения
        rotation_matrices = get_rotations(self.cfg.min_angle, self.cfg.max_angle, batch_size)

        # точки в углах исходных изображений
        vec = [[(0, 0), (width, 0), (width, height), (0, height)]]
        src_pts = torch.tensor(vec, dtype=torch.float)
        src_pts = torch.repeat_interleave(src_pts, batch_size, dim=0)

        # аффинное преобразование точек
        dst_pts = affine_transform_points(
            src_pts, rotation_matrices, (width, height), self.cfg.min_shift, self.cfg.max_shift
        )

        grid, inv_grid, homography, inv_homography, homography_pixel = calc_homography_grid(src_pts, dst_pts, batch)  # noqa: WPS236

        self.grid = grid
        self.inv_grid = inv_grid

        self.homography = homography
        self.homography_pixel = homography_pixel
        self.inv_homography = inv_homography

    def augment_and_crop(self, batch: Tensor, labels: Tensor, geometry_aug: bool):
        # цветовая аугментация
        batch_size = batch.shape[0]
        batch = adjust_brightness(batch, self.cfg.min_brightness, self.cfg.max_brightness, batch_size)
        batch = adjust_contrast(batch, self.cfg.min_contrast, self.cfg.max_contrast, batch_size)

        # геометрическая аугментация
        if geometry_aug:
            self.calculate_transform(batch)

            # для изображений используем bilinear интерполяцию
            batch = torch.grid_sampler(batch, self.inv_grid, 0, 0, align_corners=True)
            # для меток используем nearest neighbor интерполяцию, чтобы сохранить бинарные значения
            labels = torch.grid_sampler(labels.float(), self.inv_grid, 1, 0, align_corners=True)

        # центрирование и обрезка изображений и меток
        crop_size = self.cfg.crop_size

        batch, corner = center_crop(batch, crop_size)
        labels, _ = center_crop(labels, crop_size)

        # гомографию тоже центрируем и обрезаем,
        # чтобы она соответствовала размерам изображений после аугментации
        homography = crop_homography(self.homography_pixel, corner)

        return batch, labels, homography

    def warp(self, tensor: Tensor) -> Tensor:
        tensor = torch.grid_sampler(tensor, self.grid, 0, 0, align_corners=True)
        return tensor

    def inv_warp(self, tensor: Tensor) -> Tensor:
        tensor = torch.grid_sampler(tensor, self.inv_grid, 0, 0, align_corners=True)
        return tensor
