import argparse
from datetime import datetime

from dataset import *
from data_utils import *


def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default="dataset/cifar10", type=str)
    parser.add_argument("--save_dir", default="outputs")
    
    parser.add_argument("--norm_mean", default=(0.5, 0.5, 0.5), type=tuple)
    parser.add_argument("--norm_stdev", default=(0.5, 0.5, 0.5), type=tuple)
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = config()
    
    train_ds = CIFAR10Dataset(args, train=True, transform=Normalize(args.norm_mean, args.norm_stdev))
    test_ds = CIFAR10Dataset(args, train=False, transform=Normalize(args.norm_mean, args.norm_stdev))
    
    train_data, train_label = train_ds[0]
    print(train_data)