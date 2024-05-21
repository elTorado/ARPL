import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import random
import time
import os
import numpy as np
import imutil

def generate_arpl_images(net, netD, netG, iterations, trainloader, options):
    
    images = generate_images(net, netD, netG, iterations, trainloader, options)
    
    result_dir = options["result_dir"]
    images = export_images(images=images, result_dir=result_dir, dataloader=trainloader)
    print("DONE")

def generate_images(net, netD, netG, iterations, trainloader, options):
 
    net.train()
    netD.train()
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

def export_images(images, result_dir, dataloader):
    print("Initial image type:", type(images))  # Check the initial type of images
    print("Initial image shape:", getattr(images, 'shape', None))  # Safe way to get shape attribute if exists

    images = images.data.cpu().numpy()  # Assuming images is a PyTorch tensor
    print("After conversion to numpy, shape:", images.shape)  # Verify shape after conversion

    try:
        images = np.array(images).transpose((0, 2, 3, 1))
        print("After transpose, shape:", images.shape)  # This should print the new shape if transpose succeeds
    except ValueError as e:
        print("Error in transposing:", str(e))
        print("Actual shape before transposing:", images.shape)  # Output the problematic shape
        return None  # Early exit or handle error

    video_filename = make_video_filename(result_dir, dataloader, label_type='grid')
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)

    # Save the images in jpg format for visual verification
    imutil.show(images, display=False, filename=video_filename.replace('.mjpeg', '.jpg'))
    name = 'arpl{}.jpg'.format(int(time.time()))
    jpg_filename = os.path.join(result_dir, 'images', name)

    return images