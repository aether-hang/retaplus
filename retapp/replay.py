"""Exact crop-state replay, including the IPC=1 cell-slot case."""

from dataclasses import asdict, dataclass
import math
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import torch


@dataclass
class Crop:
    image: int
    top: int
    left: int
    height: int
    width: int
    flip: bool
    cell: int = -1


def draw_crop(index, height, width, generator, grid_side=0, minimum=.08):
    flip = bool(torch.rand((), generator=generator) < .5)
    if grid_side:
        if height % grid_side or width % grid_side:
            raise ValueError("A packed image must have evenly divisible cells")
        cell = int(torch.randint(grid_side**2, (), generator=generator))
        h, w = height//grid_side, width//grid_side
        return Crop(index, cell//grid_side*h, cell%grid_side*w, h, w, flip, cell)
    for _ in range(10):
        area = height*width*float(torch.empty(()).uniform_(minimum, 1, generator=generator))
        aspect = math.exp(float(torch.empty(()).uniform_(math.log(.75), math.log(4/3), generator=generator)))
        h, w = round(math.sqrt(area/aspect)), round(math.sqrt(area*aspect))
        if 0 < h <= height and 0 < w <= width:
            top = int(torch.randint(height-h+1, (), generator=generator))
            left = int(torch.randint(width-w+1, (), generator=generator))
            return Crop(index, top, left, h, w, flip)
    side = min(height, width)
    return Crop(index, (height-side)//2, (width-side)//2, side, side, flip)


def apply_crop(image, state, size):
    crop = state if isinstance(state, Crop) else Crop(**state)
    result = TF.resized_crop(image, crop.top, crop.left, crop.height, crop.width,
                            [size, size], InterpolationMode.BILINEAR, antialias=True)
    return TF.hflip(result) if crop.flip else result


def pack_cells(cells, grid_side):
    if len(cells) != grid_side**2:
        raise ValueError("The number of cells does not match the square grid")
    return torch.cat([torch.cat(list(cells[i:i+grid_side]), -1)
                      for i in range(0, len(cells), grid_side)], -2)


def encode_states(states):
    return [asdict(state) for state in states]


def replay_batch(images, states, size):
    return torch.stack([apply_crop(images.image(state["image"]), state, size) for state in states])
