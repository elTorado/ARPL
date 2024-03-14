import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import AverageMeter

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
        
        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data, True)
            logits, loss = criterion(x, y, labels)
            
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg

    return loss_all

def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, 
        trainloader, epoch=None, **options):
    print('train with confusing samples')
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()
    # What is this AverageMeter?

    net.train()
    netD.train()
    netG.train()

    torch.cuda.empty_cache()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()
        
        data, labels = Variable(data), Variable(labels)
        
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label) #fills tensor with 1's
        targetv = Variable(gan_target) #outdatd variation of making operations trackable for gradient calculations
        optimizerD.zero_grad() #reset optimizer gradients to zero
        output = netD(data) 
        errD_real = criterionD(output, targetv) #compare output tensor to target tensor using binary cross entropy los
        errD_real.backward() # backpropagation calculating gradients for the discriminator parameters with respect to the loss on real data.

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label)) #Again, create tensor but with -1s and convert to Variable (this time in one step)
        output = netD(fake.detach()) #detach somehow prevents gradients from before to flow back or so
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake #does not have influence since gradients already calculted
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad() # reset gradients (default they accumulate)
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label)) #Generator goal is to fool Discriminator into thinking its real pics, Variable tensor is filled with 1s
        output = netD(fake) #Pass fake data true D
        errG = criterionD(output, targetv) 
        """Calculates the generator's loss (errG) by comparing the discriminator's 
        predictions on the fake data against a target of real labels,
        using the binary cross-entropy loss (criterionD). 
        The goal here is to encourage the generator to produce data that 
        the discriminator classifies as real."""

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        """Call forward function of Classifier32ABN: 
        obtain features x, and their final classification y.
        --> enabled by TRUE parameter. Else it just returns y
        --> bn_label: torch tensor filled with 1s with size of number of samples inthe data batch
        dtype=torch.long means that data type is 64-bit integer
        cuda() locates in on the GPU"""
        
        errG_F = criterion.fake_loss(x).mean()
        """ This is the custom ARPL loss, takes its mean.
        Minimizing this loss encourages the generator to produce
        fake data that reduces the model's uncertainty (lower entropy) 
        about their classes, ideally making the fake data more class-discriminable 
        and closer to real data distributions."""
        
        generator_loss = errG + options['beta'] * errG_F
        """Loss is calculated by the cross-entropy loss between fake-images and true-tensor
        and the mean of the custom ARPL loss (fake data distance to class center)
        multiplied by weight for entropy loss"""
        # ---> are they sure? In the osr they say "beta" is the weight for entropy loss   
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
        """ Checks how well the classifier classifies real data to tensor of 1s w. Binary cross entropy"""

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        """options['nz'] likely stands for the dimensionality of the latent space (number of features in the noise vector), 
        and options['ns'] likely stands for the size of the noise tensor in each spatial 
        dimension if the generator expects a 4D input (like for convolutional inputs)
        --> tensor filled with values drawn from a normal distribution with mean 0 and standard deviation 1"""
        
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        "create some fresh fake data"
        
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())      
        F_loss_fake = criterion.fake_loss(x).mean()        
        total_loss = loss + options['beta'] * F_loss_fake
        """ Again, use the custom loss of binary entropic and distance loss with weights """
        total_loss.backward()
        optimizer.step()
    
        losses.update(total_loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
            .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
    
        loss_all += losses.avg

    return loss_all
