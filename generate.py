import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random
import time
import os
import numpy as np
import imutil
from PIL import Image
import pathlib
from datasets.datasets import EMNIST
import argparse


parser = argparse.ArgumentParser("Generating Images")

# Dataset
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='/home/user/heizmann/data/EMNIST/')
parser.add_argument('--outf', type=str, default='./log')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--result_dir', type=str, default='../generated_emnist')

parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)

parser.add_argument('--generate', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--number_images', type= int, help="number of images to create", default = 100)



def generate_arpl_images(netG, options):

    Data = EMNIST(options=options)
    trainloader = Data.train_loader
    iterations = options["number_images"]
    netG = get_network(netG, epoch=options.epochs, options=options)
    
    
    images = generate_images(netG, iterations, trainloader, options)
    
    result_dir = options["result_dir"]
    images = export_images(images=images, result_dir=result_dir, dataloader=trainloader)
    print("DONE")

def generate_images(netG, iterations, trainloader, options):

    
    netG.train()

    torch.cuda.empty_cache()
    
    
    images = []  # Use a list to collect tensors
    for _ in range(iterations):
        total_batches = len(trainloader)  # Total number of batches
        random_index = random.randint(0, total_batches - 1) 
        for i, (data, label) in enumerate(trainloader):
            if i == random_index:
                start_images = data        
                break
        #select a rondom batch from the trainloader
            
        noise = torch.FloatTensor(start_images.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
            start_images = start_images.cuda()

        
        #create fake data from generator
        noise = Variable(noise)
        fake = netG(noise)
        images.append(fake.cpu().detach())  # Append to list and ensure tensor is on CPU and detached
        
    # Convert list of tensors to a single tensor
    images_tensor = torch.stack(images)
    '''# NumPy array:
    images_np = images_tensor.numpy()'''
    
    return images_tensor
        
# Trajectories are written to result_dir/trajectories/
def make_video_filename(result_dir, dataloader, label_type='active'):
    trajectory_id = '{}_{}'.format(dataloader.dsf.name, int(time.time() * 1000))
    video_filename = '{}-{}-{}-{}.mjpeg'.format(label_type, trajectory_id)
    video_filename = os.path.join('trajectories', video_filename)
    video_filename = os.path.join(result_dir, video_filename)
    path = os.path.join(result_dir, 'trajectories')
    if not os.path.exists(path):
        print("Creating trajectories directory {}".format(path))
        os.mkdir(path)
    return video_filename


'''
def export_images(images, result_dir, dataloader):
    # Convert to numpy and remove singleton dimension
    images = images.data.cpu().numpy().squeeze(2)  # Shape should now be (5, 64, 32, 32)

    # Prepare directory for saving images
    images_dir = os.path.join(result_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Process each image in the batch and sequence
    for batch_index, seq in enumerate(images):
        for frame_index, frame in enumerate(seq):
            img = Image.fromarray(frame.astype('uint8'), 'L')  # Create a grayscale image
            filename = f'arpl_batch{batch_index}_frame{frame_index}_{int(time.time())}.jpg'
            img.save(os.path.join(images_dir, filename))
            
    print(f"Images are saved in: {images_dir}")
    return images  # Return the array of images (optional, depending on further needs)
'''

def export_images(images, result_dir, dataloader):
    # Convert to numpy and remove singleton dimension
    images = images.data.cpu().numpy().squeeze(2)  # Shape should now be (5, 64, 32, 32)

    # Prepare directory for saving images
    images_dir = os.path.join(result_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Process each image in the batch and sequence
    for batch_index, seq in enumerate(images):
        batch_images = []  # List to hold images for creating the grid
        for frame_index, frame in enumerate(seq):
            img = Image.fromarray(frame.astype('uint8'), 'L')  # Create a grayscale image
            filename = f'arpl_batch{batch_index}_frame{frame_index}_{int(time.time())}.jpg'
            img.save(os.path.join(images_dir, filename))
            batch_images.append(img)

        # Create and save the grid image for each batch
        if len(batch_images) > 0:
            num_cols = max(int(np.sqrt(len(batch_images))), 1)  # Ensure at least 1 column
            num_rows = (len(batch_images) + num_cols - 1) // num_cols  # Calculate rows ensuring at least 1 row
            grid_image = Image.new('L', (num_cols * frame.shape[1], num_rows * frame.shape[0]))  # Create a new empty image

            # Place images in the grid
            for index, image in enumerate(batch_images):
                row = index // num_cols
                col = index % num_cols
                grid_image.paste(image, (col * frame.shape[1], row * frame.shape[0]))
            
            # Save the grid image
            grid_filename = f'arpl_batch{batch_index}_grid_{int(time.time())}.jpg'
            grid_image.save(os.path.join(images_dir, grid_filename))
        else:
            print(f"No images to process in batch {batch_index}")

    print(f"Images and image grids are saved in: {images_dir}")
    return images  # Optionally return the array of images if needed elsewhere


def get_network(network, epoch, options):
    pth = get_pth_by_epoch(options['result_dir'], network, epoch)
    if pth:
        print("Loading {} from checkpoint {}".format(network, pth))
        network.load_state_dict(torch.load(pth))
    return network

def ensure_directory_exists(filename):
    # Assume whatever comes after the last / is the filename
    tokens = filename.split('/')[:-1]
    # Perform a mkdir -p on the rest of the path
    path = '/'.join(tokens)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
def get_pth_by_epoch(result_dir, name, epoch=None):
    checkpoint_path = os.path.join(result_dir, 'checkpoints/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    suffix = '.pth'
    if epoch is not None:
        suffix = 'epoch_{:04d}.pth'.format(epoch)
    files = [f for f in files if '{}_epoch'.format(name) in f]
    if not files:
        return None
    files = [os.path.join(checkpoint_path, fn) for fn in files]
    files.sort(key=lambda x: os.stat(x).st_mtime)
    return files[-1]