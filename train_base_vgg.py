""" This file is for training original model without routing modules.
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import os
import shutil
import argparse
import time
import logging

from prepare_data import *

import numpy as np

import warnings
warnings.filterwarnings('ignore')

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#from utils import gradCheck,nowTime

import vgg_base

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['train', 'test'], default='train')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset type')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--iters', default=64000, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-iter', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=1, type=float,
                        help='initial learning rate')
    parser.add_argument('--step-epoch', type = int, default = 25,
                        help='lr adjust interval')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 100)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--eval-every', default=1000, type=int,
                        help='evaluate model every (default: 1000) iterations')
    parser.add_argument('--cuda', type = bool, default = True)
    parser.add_argument('--gpu', default = '1',
                        help='GPU id')
    parser.add_argument('--logname', type = str, default = '',
                        help='')

    args = parser.parse_args()
    return args

N = 50000

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.cuda = args.cuda and torch.cuda.is_available()

    save_path = args.save_path = os.path.join(args.save_folder, 'cifar10_base' + args.logname)
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    # config logging file
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training ... ')
        run_training(args)

    elif args.cmd == 'test':
        logging.info('start evaluating with checkpoints from {}'.format(
            args.resume))
        test_model(args)

def run_training(args):
    # create model
    #model = vgg.bayesian_vgg16_bn()
    model = vgg_base.cifar_vgg16_bn_sp_dropout()
    model = model.cuda()
    print(model)

    best_prec1 = 0

    logging.info("This is for base vgg training, with: \n"
                    "Model: VGG16 with BN and \n"
                    "       different dropout rate across layers,\n"
                    "Batch Size: {}, \n"
                    "Optimizer: SGD, \n"
                    "Criterion: Cross Entropy, \n"
                    "LearnRate: from {}, and divided by {} every {} epochs after {}th epoch. \n".format(
                    args.batch_size,args.lr,1/args.step_ratio, args.step_epoch, 2 * args.step_epoch
                    ))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

     # define loss function (criterion)
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                args.lr,
                                momentum=0.9,
                                weight_decay=0.0005)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    for i in range(args.start_iter, args.iters):
        model.train()
        adjust_learning_rate(args, optimizer, i)

        input, target = next(iter(train_loader))

        # measuring data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input,target = input.cuda(),target.cuda()
        input_var,target_var = Variable(input).cuda(), Variable(target).cuda()

        optimizer.zero_grad()

        # calculate the output and loss
        output = model(input_var)
        loss = criterion(output, target)

        # measure accuracy, and record loss and accuracy
        prec1, = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        if i % args.print_freq == 0:
            logging.info("Iter: [{0:5}/{1}] "
                         "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                         "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                         "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                         "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                            i,
                            args.iters,
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            top1=top1)
            )
            losses.reset()
            top1.reset()

            # evaluate every 1000 steps
        if (i % args.eval_every == 0 and i > 0) or (i == args.iters - 1):
            prec1 = validate(args, test_loader, model, criterion)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            checkpoint_path = os.path.join(args.save_path,
                                           'checkpoint_{:05d}.pth.tar'.format(
                                               i))
            save_checkpoint({
                'iter': i,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            },
                is_best, filename=checkpoint_path)
            shutil.copyfile(checkpoint_path, os.path.join(args.save_path,
                                                          'checkpoint_latest'
                                                          '.pth.tar'))

def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluation mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        #target = target.cuda(async=True)
        if args.cuda:
            input,target = input.cuda(),target.cuda()

        input_var = Variable(input, volatile=True)
        target_var = Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, = accuracy(output.data, target, topk=(1,))
        top1.update(prec1.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) or (i == len(test_loader) - 1):
            logging.info(
                'Test: [{}/{}]\t'
                'Time: {batch_time.val:.4f}({batch_time.avg:.4f})\t'
                'Loss: {loss.val:.3f}({loss.avg:.3f})\t'
                'Prec@1: {top1.val:.3f}({top1.avg:.3f})\t'.format(
                    i, len(test_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def test_model(args):
    # create model
    model = vgg.bayesian_vgg16_bn()
    model = model.cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info('=> loading checkpoint `{}`'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_iter = checkpoint['iter']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (iter: {})'.format(
                args.resume, checkpoint['iter']
            ))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    cudnn.benchmark = False
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    validate(args, test_loader, model, criterion)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path,
                                               'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, _iter):
    """divide lr by 10 at 32k and 48k """
    step_epoch = args.step_epoch
    single_step = 25 * 50000 / args.batch_size

    if _iter < single_step:
        lr = args.lr
    elif single_step <= _iter < single_step * 2:
        lr = args.lr * (args.step_ratio ** 0)
    elif single_step * 2 <= _iter < single_step * 3:
        lr = args.lr * (args.step_ratio ** 1)
    elif single_step * 3 <= _iter < single_step * 4:
        lr = args.lr * (args.step_ratio ** 2)
    else:
        lr = args.lr * (args.step_ratio ** 3)

    if _iter % args.eval_every == 0:
        logging.info('Iter [{}] learning rate = {}'.format(_iter, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
