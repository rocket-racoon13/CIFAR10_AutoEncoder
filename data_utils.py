from typing import Sequence, Tuple, List

import torch

from torch.utils.data.dataloader import default_collate


class Normalize:
    def __init__(self, mean: Sequence, stdev: Sequence):
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.stdev = torch.FloatTensor(stdev).view(-1, 1, 1)
        print(self.mean, self.stdev)
        
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