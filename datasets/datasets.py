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

'''
    This file contains the dataset classes  for ImageNet and Emnist. 
    There are also other implementations conserved from the older implementation
    such as MNIST. 

'''


def pad_tensor(img, target_size=(32, 32)):
    """Pads a given tensor to the target size with constant values.

    Args:
        img (torch.Tensor): The input image tensor to be padded.
        target_size (tuple, optional): The target size (width, height) to pad the image to. Defaults to (32, 32).

    Returns:
        torch.Tensor: The padded image tensor.
    """   
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
    """
    EMNIST Dataset class.
    This class loads the EMNIST dataset, applies necessary transformations, and creates DataLoaders for training, validation, and testing.
    It also provides unique getter methods for itam and all labels and a transform method. 

    Args:
        val (bool, optional): If True, loads the validation set. Defaults to True.
        test (bool, optional): If True, loads the test set. Defaults to True.
        **options (dict): Additional options for the dataset.

    Attributes:
        workers (int): Number of worker processes for loading data.
        which_set (str): Training set being used, default is "train".
        batch_size (int): Batch size for the DataLoader.
        dataset_root (str): Root directory of the Imagenet dataset.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        num_classes (int): Number of classes in the dataset.
        traindata (torchvision.datasets.EMNIST): Training dataset.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valdata (torchvision.datasets.EMNIST): Validation dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        letters (CustomEMNIST): Test dataset containing letters.
        out_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    
    """
 
    
    # We only need the digits from the mnist split for the ARPL implementation, hence there is no need to create any logic around the letters.
    def transform(x):
        """Transform function to pad and transpose the tensor. """
        x = pad_tensor(x)
        x = x.transpose(2,1)
        
        return x
    
    def get_labels(dataloader):
            """
                Get unique labels from the dataloader
            
                Return:
                unique_labels (set): set of all the labels in the dataloader  
            """
        
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
            
            self.out_loader = torch.utils.data.DataLoader(
            self.letters, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.workers,  pin_memory=self.pin_memory,
            )               
                
        def __getitem__(self, index):
            """
                Get item from the dataset at the specified index.

                Args:
                    index (int): Index of the item to be fetched.

                Returns:
                    tuple: (image, target) where image is the transformed image and target is the label.
            """           
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
  
class ImageNet(torch.utils.data.dataset.Dataset):
    """ 
        ImageNet dataset class
        Custom Dataset for loading ImageNet data from a CSV file.
        Filters out negative labels and loads images from the specified path.
        It also applies transformations to the images if provided.

        Args:
            csv_file (str): Path to the CSV file containing image paths and labels.
            imagenet_path (str): Path to the ImageNet images directory.
            transform (callable, optional): Optional transform to be applied on an image.

        Attributes:
            dataset (pd.DataFrame): DataFrame containing image paths and labels.
            imagenet_path (Path): Path object for the ImageNet images directory.
            transform (callable): Transform to be applied on an image.
            label_count (int): Number of unique labels in the dataset.
            unique_classes (np.ndarray): Sorted array of unique classes in the dataset.
    """
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