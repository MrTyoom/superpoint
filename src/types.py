import typing as tp

import torch
from numpy import typing as npt


Tensor: tp.TypeAlias = torch.Tensor
Array: tp.TypeAlias = npt.NDArray
TupleInt: tp.TypeAlias = tuple[int, int]
