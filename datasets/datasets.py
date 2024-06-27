import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST
import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
from utils import mkdir_if_missing

def pad_tensor(img, target_size=(32, 32)):
    # Calculate padding for each side
    height_pad = (target_size[1] - img.size(1)) // 2
    width_pad = (target_size[0] - img.size(2)) // 2

    # Apply padding
    # The padding format is (left, right, top, bottom)
    padded_img = F.pad(img, (width_pad, width_pad, height_pad, height_pad), mode='constant', value=0)
    return padded_img


class CustomEMNIST(torch.utils.data.dataset.Dataset):
    def __init__(self, root, transform=None):
        self.emnist = torchvision.datasets.EMNIST(
            root=root,
            split='letters',
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.emnist)

    def __getitem__(self, index):
        # Retrieve the original data but ignore its label
        data, _ = self.emnist[index]
        # Return the data with label -1
        return data, -1

class EMNIST(torch.utils.data.dataset.Dataset):
    
    ''' IS ZERO PADDING NECESSARY AS WE CAN CHOOSE AN IMAGE SIZE??'''
    
    # We only need the digits from the mnist split for the ARPL implementation, hence there is no need to 
    # create any logic around the letters.
    
    def transform(x):
        
        x = pad_tensor(x)
        x = x.transpose(2,1)
        
        return x
    
    def get_labels(dataloader):
            unique_labels = set()
            for data in dataloader:
                inputs, labels = data
                unique_labels.update(labels.numpy())
            return unique_labels
                
    def __init__(self, val = True, test = True,**options):
        
        options=options['options']

        self.workers = 8
        self.which_set = "train"
        self.batch_size = options['batch_size']
        self.dataset_root = os.path.join(options['dataroot'])
        self.pin_memory = True if options['use_gpu'] else False
        self.num_classes = 10
          
        print(" DATASET ROOR IS :", self.dataset_root) 
        
        ############## TRAIN DATA ########################
        
        self.traindata = torchvision.datasets.EMNIST(
            root=self.dataset_root,
            train=True,
            download=True,
            split="mnist",
            transform=transforms.Compose([transforms.ToTensor(), EMNIST.transform])
        )
        
        self.train_loader = torch.utils.data.DataLoader(
            self.traindata, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=self.pin_memory,
        )
        
        print("TRAINING LABELS: ", EMNIST.get_labels(self.train_loader))

        ############## VALIDATION DATA ########################
        
        if val:
            self.valdata = torchvision.datasets.EMNIST(
                root=self.dataset_root,
                train=False,
                download=True,
                split="mnist",
                transform=transforms.Compose([transforms.ToTensor(), EMNIST.transform])
            )
            print("TEST LABELS: ", EMNIST.get_labels(self.test_loader))
            
            self.test_loader = torch.utils.data.DataLoader(
                    self.valdata, batch_size=self.batch_size, shuffle=False,
                    num_workers=self.workers, pin_memory=self.pin_memory,
                    )
        
        ############## TEST DATA ########################
        
        if test:
            self.letters = CustomEMNIST(
                root=self.dataset_root,
                transform=transforms.Compose([transforms.ToTensor(), EMNIST.transform  ])
            )
            print("OPEN SET LABELS: ", EMNIST.get_labels(self.out_loader))
            self.out_loader = torch.utils.data.DataLoader(
            self.letters, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.workers,  pin_memory=self.pin_memory,
            )
    
        

        

               
                
                
        def __getitem__(self, index):
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
  
class ImageNet(torch.utils.data.dataset.Dataset):
    
    def __init__(self, csv_file, imagenet_path, transform = None):
        
        self.dataset = pd.read_csv(csv_file, header=None)
        
        # for GAN training we dont want any negatives as we want to create synthetic negatives from known classes
        self.dataset = self.dataset[self.dataset[1] != -1]
        
        self.imagenet_path = Path(imagenet_path)
        self.transform = transform
        self.label_count = len(self.dataset[1].unique())
        self.unique_classes = np.sort(self.dataset[1].unique())
        
    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.dataset)

    def __getitem__(self, index):
        """ Returns a tuple (image, label) of the dataset at the given index. If available, it
        applies the defined transform to the image. Images are converted to RGB format.

        Args:
            index(int): Image index

        Returns:
            image, label: (image tensor, label tensor)
        """
        if torch.is_tensor(index):
            index = index.tolist()

        jpeg_path, label = self.dataset.iloc[index]
        image = Image.open(self.imagenet_path / jpeg_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # convert int label to tensor
        label = torch.as_tensor(int(label), dtype=torch.int64)
        return image, label

  
  
        
class MNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['dataroot'], 'mnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = MNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = MNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10
     


__factory = {
    'mnist': MNIST,
    'kmnist': KMNIST,
}

def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**options)