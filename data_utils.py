from typing import Sequence

import torch


class Normalize:
    def __init__(self, mean: Sequence, stdev: Sequence):
        self.mean = torch.FloatTensor(mean).view(-1, 1, 1)
        self.stdev = torch.FloatTensor(stdev).view(-1, 1, 1)
        print(self.mean, self.stdev)
        
    def __call__(self, image: torch.Tensor):
        output = (image - self.mean) / self.stdev
        return output


def custom_collator():
    pass