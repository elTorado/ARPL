import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib
from torchvision import transforms
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from models import gan
from models.models import classifier32, classifier32ABN
from datasets.datasets import EMNIST, ImageNet
from datasets.osr_dataloader import MNIST_OSR
from utils import Logger, save_networks, load_networks
from core import train_cs, test, save_network

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='/home/user/heizmann/data/EMNIST/')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')
parser.add_argument('--protocol', type=int, default=2, help='imagenet protocol')


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


def main_worker(options):
    
    #options['dataroot'] = '/home/deanheizmann/dataset/emnist'
    options['dataroot'] = '/home/user/heizmann/dataset/emnist'
    
    imagenet_path = '/local/scratch/datasets/ImageNet/ILSVRC2012/'
    train_file = 'protocols/p{}_train.csv'

    
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        options.update({'use_gpu':  True})
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")
    # Dataset
    print("{} Preparation".format(options['dataset']))
    
    if 'emnist' in options['dataset']:
        Data = EMNIST(options=options)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    if 'imagenet' in options['dataset']:
        
        train_tr = transforms.Compose(
            [transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])

        val_tr = transforms.Compose(
            [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
        
        train_file = pathlib.Path(train_file.format(options.protocol)),
        imagenet_path= imagenet_path
        
        trainloader = ImageNet(
            csv_file=train_file,
            imagenet_path=imagenet_path,
            transform=train_tr
        )
        

    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    if options['cs']:
        net = classifier32ABN(num_classes=options['num_classes'])
    else:
        net = classifier32(num_classes=options['num_classes'])
    feat_dim = 128

    if options['cs']:
        print("Creating GAN")
        nz, ns = options['nz'], 1

        if 'imagenet' in options['dataset']:
            netG = gan.Generator256(1, nz, 64, 1)
            netD = gan.Discriminator256(1, 1, 64)
            fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
            criterionD = nn.BCELoss()
            
        else: 
            netG = gan.Generator32(1, nz, 64, 1)
            netD = gan.Discriminator32(1, 1, 64)
            fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
            criterionD = nn.BCELoss()

    # Loss
    options.update(
    feat_dim=feat_dim,
    use_gpu=use_gpu
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    

    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], 50, options['item'], options['cs'])
    else:
        file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], options['item'], options['cs'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    
    if options['dataset'] == 'tiny_imagenet':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
    if options['cs']:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120])

    start_time = time.time()

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        if options['cs']:
            loss_all, netG = train_cs(net, netD, netG, criterion, criterionD,
                optimizer, optimizerD, optimizerG,
                trainloader, epoch=epoch, **options)

            save_network(netG, epoch=options["max_epoch"], result_dir=options["result_dir"])
       
        if options['stepsize'] > 0: scheduler.step()

    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    print("DONE")

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()
    
    from split import splits_2020 as splits


    known = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    unknown = [-1]

    options.update(
        {
            'item':     "emnist",
            'known':    known,
            'unknown':  unknown,
            'img_size': img_size
        }
    )

    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(options['outf'], 'results', dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    file_name = "emnist" + '.csv'
    
    main_worker(options)
    
    '''res['unknown'] = unknown
    res['known'] = known
    results["EMNIST"] = res
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(dir_path, file_name))
    print("saved csv to: ", os.path.join(dir_path, file_name))'''