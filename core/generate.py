import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import random
import time
import os
import numpy as np
import imutil

def generate_arpl_images(net, netD, netG, iterations, trainloader, **options):
    
    images = generate_images(net, netD, netG, iterations, trainloader, **options)
    
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
            images.append(fake.cpu().detach())  # Append to list and ensure tensor is on CPU and detached

        
        #create fake data from generator
        noise = Variable(noise)
        fake = netG(noise)
        images.append(fake)
        
    # Convert list of tensors to a single tensor
    images_tensor = torch.stack(images)
    # NumPy array:
    images_np = images_tensor.numpy()
    
    return images_np
        
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
    images = images.data.cpu().numpy()
    images = np.array(images).transpose((0,2,3,1))
    video_filename = make_video_filename(result_dir, dataloader, label_type='grid')
     # Save the images in npy/jpg format as input for the labeling system
    trajectory_filename = video_filename.replace('.mjpeg', '.npy')
    np.save(trajectory_filename, images)
    imutil.show(images, display=False, filename=video_filename.replace('.mjpeg', '.jpg'))

    # Save the images in jpg format to display to the user
    name = 'arpl{}.jpg'.format(int(time.time()))
    jpg_filename = os.path.join(result_dir, 'images', name)
    return images