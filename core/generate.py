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

import os
import numpy as np
import torch
from PIL import Image
import time

def export_images(images, result_dir, dataloader):
    images = images.data.cpu().numpy().squeeze(2)  # Shape should now be (5, 64, 32, 32)

    images_dir = os.path.join(result_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for batch_index, seq in enumerate(images):
        batch_images = [] 
        for frame_index, frame in enumerate(seq):
            img = Image.fromarray(frame.astype('uint8'), 'L')  # Create a grayscale image
            filename = f'arpl_batch{batch_index}_frame{frame_index}_{int(time.time())}.jpg'
            img.save(os.path.join(images_dir, filename))
            batch_images.append(img)

        # Create and save the grid image for each batch
        num_cols = int(np.sqrt(len(batch_images)))  
        num_rows = (len(batch_images) + num_cols - 1) // num_cols  
        grid_image = Image.new('L', (num_cols * frame.shape[2], num_rows * frame.shape[1]))  
        
        for index, image in enumerate(batch_images):
            row = index // num_cols
            col = index % num_cols
            grid_image.paste(image, (col * frame.shape[2], row * frame.shape[1]))
        
        # Save the grid image
        grid_filename = f'arpl_batch{batch_index}_grid_{int(time.time())}.jpg'
        grid_image.save(os.path.join(images_dir, grid_filename))

    print(f"Images and image grids are saved in: {images_dir}")
    return images  # Optionally return the array of images if needed elsewhere
