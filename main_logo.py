import argparse
import builtins
import math
import os
import random
import shutil
import warnings


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

import utils
from logo.loader import load_datasets
import logo.builder as builder
import trainers

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--store-path', default='', type=str,
                    help='path to store the results')
parser.add_argument('--save-step', default=1, type=int,
                    help='step for saving checkpoints')

parser.add_argument('--dataset', default='imagenet100', type=str,
                    help='dataset to pretrain')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--lr_list', default=[0.03,0.03], type = float, nargs='+',
                    help='list of initial learning rate')
parser.add_argument('--fea-dim', default=128, type=int,
                    help='feature dimension (default: 128)')

#crop size setting
parser.add_argument('--mc', action='store_true',
                    help='use multi-crop training')
parser.add_argument("--global-scale", type=float, default=[0.4, 1.], nargs="+",
                    help="argument in RandomResizedCrop ")
parser.add_argument("--local-scale", type=float, default=[0.05, 0.4], nargs="+",
                    help="argument in RandomResizedCrop ")


# moco setting:
parser.add_argument('--moco-k', default=4096, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

#LoGo setting
parser.add_argument('--logo', action='store_true',
                    help='use LoGo-style multi-crops structure')
parser.add_argument('--reg-dim', default = 256, type = int,
                    help='dim of hidden layers of regressor')
parser.add_argument('--cr', default = 0.015, type = float)


def main():
    args = parser.parse_args()
    if not os.path.exists(args.store_path):
        os.makedirs(os.path.join(args.store_path,'record'))
        os.makedirs(os.path.join(args.store_path,'checkpoint'))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.logo:
        args.mc = True

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))

    model = builder.MoCo_LoGo(
        args.arch, args.fea_dim, args.moco_k, args.moco_m, args.moco_t, args.dataset
    )

    if (args.mc and args.logo):
        regressor = builder.Regressor_softplus(args.fea_dim, args.reg_dim)
    print(model)

    if args.logo:
        args.lr_list[0] = args.lr_list[0] * args.batch_size / 256
        args.lr_list[1] = args.lr_list[1] * args.batch_size / 256
    else:
        args.lr_list[0] = args.lr_list[0] * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(args.gpu)
            if args.mc and args.logo:
                regressor = regressor.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            if args.mc and args.logo:
                regressor.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        if args.mc and args.logo:
            regressor = regressor.cuda(args.gpu)
            print(regressor)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    #print(args.lr_list)

    optimizer = torch.optim.SGD(model.parameters(), args.lr_list[0],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.logo:
        optimizer_reg = torch.optim.SGD(regressor.parameters(), args.lr_list[1],
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        optimizer_logo = torch.optim.SGD(model.parameters(), args.lr_list[0],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.logo:
                regressor.load_state_dict(checkpoint['regressor'])
                optimizer_reg.load_state_dict(checkpoint['optimizer_reg'])
                optimizer_logo.load_state_dict(checkpoint['optimizer_logo'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_dataset = load_datasets(args.dataset, args.data, args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args, args.lr_list[0])
        if args.mc and args.logo:
            adjust_learning_rate(optimizer_logo, epoch, args, args.lr_list[0])
            #adjust_learning_rate(optimizer_reg, epoch, args, args.lr_list[1])
            #cr = adjust_clipping_rate(epoch, args)
            cr = args.cr
            ls_model = [model, regressor]
            ls_optimizer = [optimizer, optimizer_logo, optimizer_reg]
            trainers.train_moco(train_loader, ls_model, criterion, ls_optimizer, epoch, args, cr)
        else:
            ls_optimizer = [optimizer]
            ls_model = [model]
            trainers.train_moco(train_loader, ls_model, criterion, ls_optimizer, epoch, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0 and (epoch+1)%args.save_step == 0):
            if args.mc and args.logo :
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'regressor': regressor.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'optimizer_reg': optimizer_reg.state_dict(),
                    'optimizer_logo': optimizer_logo.state_dict()
                }, is_best=False, filename=os.path.join(args.store_path,'checkpoint',
                                                        '{}_checkpoint_logo_{:04d}_epoch.pth.tar'.format('moco',epoch)))
            else:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.store_path,'checkpoint',
                                                        '{}_checkpoint_{:04d}_epoch.pth.tar'.format('moco',epoch)))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, args, lr):
    """Decay the learning rate based on schedule"""
    init_lr = lr
    if args.cos:  # cosine lr schedule
        lr *= .5*(1 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = lr


def adjust_clipping_rate(epoch, args):
    """Increase the gradient clipping rate based on schedule"""

    cr = args.cr*0.5 * (1. + math.cos(math.pi * (args.epochs-epoch) / args.epochs))
    return cr


if __name__ == '__main__':
    main()
