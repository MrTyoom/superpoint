from torch import nn

from src.train_utils.d2s import DepthToSpace


def get_heatmap(semi):
    semi_soft = nn.functional.softmax(semi, dim=1)
    nodust = semi_soft[:, :-1, :, :]

    depth_to_space = DepthToSpace(8)
    heatmap = depth_to_space(nodust)
    return heatmap
