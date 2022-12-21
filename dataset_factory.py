import os
from torchvision.datasets import CIFAR100, CIFAR10


def create_dataset(name, 
                   data_path = "./data",
                   is_training=False,
                   transform=None,
                   target_transform=None,
                   download=True):
    
    if name == "CIFAR10":
        ds = CIFAR10(root=data_path,
                     train=is_training,
                     transform=transform,
                     target_transform=target_transform,
                     download=download)
    
    elif name == "CIFAR100":
        ds = CIFAR100(root=data_path,
                     train=is_training,
                     transform=transform,
                     target_transform=target_transform,
                     download=download)
    
    return ds
    