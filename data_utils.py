from PIL import Image
from typing import Sequence, Tuple, List

import torch
import torchvision.transforms as transforms


class Normalize:
    def __init__(self, mean: Sequence, stdev: Sequence):
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.stdev = torch.FloatTensor(stdev).view(-1, 1, 1)
        
    def __call__(self, image: torch.Tensor):
        output = (image - self.mean) / self.stdev
        return output


def custom_collator(batch: List[Tuple[torch.Tensor, int]]): # [(tensor, int), (tensor, int), (tensor, int)] -> [tensor, int]
    r"""Puts each data field into a tensor with outer dimension batch size"""
    image_list = [item[0] for item in batch]
    label_list = [torch.tensor(item[1]) for item in batch]
    batched_image = torch.stack(image_list, dim=0)
    batched_label = torch.stack(label_list, dim=0)
    return [batched_image, batched_label]


def read_and_convert_image_to_pt(image_dir) -> torch.Tensor:
    """
    Loads an image with PIL and converts to torch.Tensor in the CHW sequence.
    If the image is in the RGBA format, the alpha channel is deleted.
    """
    image = Image.open(image_dir)
    image_pt = torch.as_tensor(np.array(image, copy=True))   # HWC
    if image_pt.dim() == 2:
        image_pt = image_pt.unsqueeze(dim=-1)
    if image_pt.size(-1) == 4:
        image_pt = image_pt[:, :, :3]   # RGBA to RGB
    image_pt = image_pt.permute((2, 0, 1))   # CHW
    return image_pt


def eval_transform(args, image_dir):
    resizer = transforms.Resize((args.image_height, args.image_width))
    grayscaler = transforms.Grayscale(args.image_channel)
    
    image_pt = read_and_convert_image_to_pt(image_dir)
    image_pt = resizer(image_pt)
    image_pt = grayscaler(image_pt) if image_pt.size(0) == 3 else image_pt
    image_pt = 255 - image_pt   # invert background color # https://medium.com/@krishna.ramesh.tx/training-a-cnn-to-distinguish-between-mnist-digits-using-pytorch-620f06aa9ffa
    image_pt = normalize(image_pt / 255, 0.5, 0.5)
    if args.model_type.lower() == "ann":
        image_pt = image_pt.flatten()
    
    return image_pt


def normalize(
    input: torch.Tensor,
    mean: float,
    std: float
):
    output = (input-mean)/std
    return output