import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
from torch.utils.data import DataLoader
from dataset_factory import create_dataset

# urllib.error.URLError
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

print(torch.__version__)

num_classes = 10
batch_size = 32
epoch = 30
log_dir = "log"
dataset_dir = "./data/cifar10"
dataset_name = "CIFAR10"

def train():
    # writer = SummaryWriter(log_dir)
    
    train_dataset = create_dataset(dataset_name, 
                                   data_path=dataset_dir, 
                                   is_training=True, 
                                   transform=transforms.ToTensor(),
                                   download=True)
    
    test_dataset = create_dataset(dataset_name,
                                  data_path=dataset_dir,
                                  is_training=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    
    return


if __name__ == "__main__":
    train()