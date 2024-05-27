import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter
import pathlib
import os
from vast.tools import set_device_gpu, set_device_cpu, device

def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, 
        trainloader, epoch=None, **options):
    print('train with confusing samples')
    
    # setup device
    if options['use_gpu'] is not None:
        set_device_gpu(index=options['gpu'] )
        print(" ============== GPU Selected! =============")
    else:
        print("No GPU device selected, training will be extremely slow")
        set_device_cpu()
    
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    netD.train()
    netG.train()
    net = device(net)
    netD = device(netD)
    netG = device(netG)
    
    torch.cuda.empty_cache()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        

            
        data, labels = Variable(data), Variable(labels)
        
        data = device(data)
        labels = device(labels)
        gan_target = device(gan_target)
        
        
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()

        noise = Variable(noise)
        noise = device(noise)
        
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + options['beta'] * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))


        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        noise = Variable(noise)
        noise = device(noise)
        
        fake = netG(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer.step()
    
        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
            .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
    
    
        loss_all += losses.avg


    return loss_all, netG

def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def save_network(network, epoch, result_dir):
    weights = network.state_dict()
    filename = '{}/checkpoints/{}_epoch_{:04d}.pth'.format(result_dir, "netG", epoch)
    print(f"Attempting to save weights to: {filename}")  # Shows the full path attempting to save to
    
    # Ensure the directory exists
    ensure_directory_exists(filename)
    
    # Check if the directory is writable
    directory = os.path.dirname(filename)
    if os.access(directory, os.W_OK):
        print("Directory is writable")
    else:
        print("Directory is not writable")
    
    # Attempt to save the file
    try:
        torch.save(weights, filename)
        print("Save successful")
    except Exception as e:
        print(f"Failed to save: {e}")

    # Additional debug to check the directory and permissions after the fact
    print(f"Contents of directory {directory}:")
    print(os.listdir(directory))  # List files to see what's actually in there
    


