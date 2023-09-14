import argparse
from datetime import datetime

from dataset import *
from data_utils import *


def config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=77)
    
    parser.add_argument("--data_dir", type=str, default="dataset/cifar10")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--model_name", type=str)
    
    parser.add_argument('--image_width', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=32)
    parser.add_argument('--image_channel', type=int, default=3)
    
    parser.add_argument("--norm_mean", type=tuple, default=(0.5, 0.5, 0.5))
    parser.add_argument("--norm_stdev", type=tuple, default=(0.5, 0.5, 0.5))
    
    parser.add_argument('--conv_channels', type=list, default=[8, 16])
    parser.add_argument('--kernel_size', type=list, default=[5, 5])
    parser.add_argument('--stride', type=int, default=[1, 1])
    parser.add_argument('--padding', type=list, default=[1, 1])
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default="lambdaLR")
    
    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=500)
    
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--predict', action="store_true")
    parser.add_argument('--no_cuda', action="store_true")
    
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = config()
    
    train_ds = CIFAR10Dataset(args, train=True, transform=Normalize(args.norm_mean, args.norm_stdev))
    test_ds = CIFAR10Dataset(args, train=False, transform=Normalize(args.norm_mean, args.norm_stdev))
    
    train_data, train_label = train_ds[0]
    print(train_data)