import os, sys
import time
import torch
import socket
import argparse
import subprocess
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from typing import Tuple
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


import subprocess, shlex, shutil, io, os
import torch, torchvision, glob, tqdm, scipy, gc, time
from time import strftime, localtime
from datetime import timedelta
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils import data
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

import clean_cornets  # custom networks based on the CORnet family from di carlo lab
import ds2  # custom module for datasets
import argparse

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_choice', default='z',
                    help='z for cornet Z,  s for cornet S')
parser.add_argument('--img_path', default='../../ecoset',
                    help='path to ImageNet folder that contains train and val folders')
parser.add_argument('--wrd_path', default= 'wordsets_FR',
                    help='path to word folder that contains train and val folders')
parser.add_argument('--save_path', default='save/',
                    help='path for saving ')
parser.add_argument('--output_path', default='activations/',
                    help='path for storing activations')
parser.add_argument('--restore_file', default='save/save_pre_z_49_full_nomir.pth.tar',
                    help='name of file from which to restore model (ought to be located in save path, e.g. as save/cornet_z_epoch25.pth.tar)')
parser.add_argument('--img_classes', default=565,
                    help='number of image classes')
parser.add_argument('--wrd_classes', default=10000,
                    help='number of word classes')
parser.add_argument('--num_train_items', default=2500,
                    help='number of training items in each category')
parser.add_argument('--num_val_items', default=50,
                    help='number of validation items in each category')
parser.add_argument('--num_workers', default=10,
                    help='number of workers to load batches in parallel')
parser.add_argument('--mode', default='pre',
                    help='pre for pre-schooler mode, lit for literate mode')
parser.add_argument('--max_epochs_pre', default=80, type=int,
                    help='number of epochs to run as pre-schooler - training on images only')
# parser.add_argument('--max_epochs_lit', default=30, type=int,
#                     help='number of epochs to run as literate - training on images and words')
parser.add_argument('--batch_size', default=100, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.01, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_schedule', default='StepLR')
parser.add_argument('--step_size', default=20, type=int,
                    help='after how many epoch learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

FLAGS, _ = parser.parse_known_args()


# useful
def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def train(mode = FLAGS.mode, restore_path=None, save_path=FLAGS.save_path, plot=0, show=0):
    start_time = time.time()

    global net
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = "cpu"
    torch.backends.cudnn.benchmark = True

    if mode == 'pre':
        # Datasets and Generators
        ##############################################
        # transform = transforms.Compose(
        #     [transforms.RandomResizedCrop(224),
        #       transforms.ToTensor(),
        #       transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                           std=[0.229, 0.224, 0.225])])

        # Initialize Datasets
        # root_imagenet = os.path.join(os.environ["DSDIR"], "imagenet", "RawImages")
        # print(root_imagenet)
        # print(sys.version_info)
        # train_imgset = datasets.ImageNet(root=root_imagenet, split='train', download=False, transform=transform)
        # val_imgset = datasets.ImageNet(root=root_imagenet, split='val', download=False, transform=transform)
        ##############################################

        train_imgset = ds2.ImageDataset(data_path=FLAGS.img_path, folder='train')
        training_gen = data.DataLoader(train_imgset, batch_size=FLAGS.batch_size, shuffle=True,
                                       num_workers=FLAGS.num_workers)
        del train_imgset
        gc.collect()

        val_imgset = ds2.ImageDataset(data_path=FLAGS.img_path, folder='val')
        validation_gen = data.DataLoader(val_imgset, batch_size=FLAGS.num_val_items, shuffle=False,
                                         num_workers=FLAGS.num_workers)

        # variables, labels, prints, and titles for plots
        cat_scores = np.zeros((FLAGS.max_epochs_pre, FLAGS.img_classes))

        trainloss, valloss = [], []
        max_epochs = FLAGS.max_epochs_pre
        shift_epoch = 0
        print_save = 'saving pre-schooler model'

        # Model
        net = clean_cornets.CORnet_Z_tweak()

    # use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    # transfer model to device
    net.to(device)

    exec_time = secondsToStr(time.time() - start_time)
    print('execution time so far: ', exec_time)

    # Build loss function, model and optimizer.
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    """
    train
    """

    # Loop over epochs
    for epoch in range(max_epochs):

        gc.collect()
        # Training
        print('epoch', shift_epoch + epoch)
        # scheduler.step()

        if FLAGS.save_path != None:
            # Save model
            ckpt_data = {}
            ckpt_data['epoch'] = shift_epoch + epoch
            ckpt_data['state_dict'] = net.state_dict()
            ckpt_data['optimizer'] = optimizer.state_dict()
            print(print_save)
            torch.save(ckpt_data, FLAGS.save_path + 'save_' + mode + '_' + FLAGS.model_choice + '_' + str(
                shift_epoch + epoch) + '_full_nomir.pth.tar')
            np.save(save_path + 'cat_scores_' + mode + '_' + FLAGS.model_choice + '_full_nomir.npy', cat_scores)
            np.save(save_path + 'trainloss_' + mode + '_' + FLAGS.model_choice + '_full_nomir.npy', np.array(trainloss))
            np.save(save_path + 'valloss_' + mode + '_' + FLAGS.model_choice + '_full_nomir.npy', np.array(valloss))

        # Validation
        with torch.set_grad_enabled(False):
            cat_index = 0
            for local_batch_val, local_labels_val in validation_gen:
                print('cat_index', cat_index)

                # Transfer to GPU
                local_batch_val, local_labels_val = local_batch_val.to(device), local_labels_val.to(device)

                # Model computations
                v1, v2, v4, it, h, pred_val = net(local_batch_val)

                scores = Acc(pred_val, local_labels_val)
                print('category accuracy scores', scores)
                cat_scores[shift_epoch + epoch, cat_index] = scores
                print('cat_scores[shift_epoch + epoch, cat_index]', cat_scores[shift_epoch + epoch, cat_index])
                exec_time = secondsToStr(time.time() - start_time)
                print('execution time so far: ', exec_time)
                cat_index += 1
                torch.cuda.get_device_name(0)
                # Compute loss.
                loss_val = criterion(pred_val, local_labels_val)
                valloss += [loss_val.item()]

        for local_batch, local_labels in training_gen:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            torch.cuda.empty_cache()

            # Forward pass.
            v1, v2, v4, it, h, pred = net(local_batch)

            # Compute loss.
            loss = criterion(pred, local_labels)
            trainloss += [loss.item()]
            print('epoch :', shift_epoch + epoch, ', loss: ', loss.item())
            end_time = time.time()
            exec_time = secondsToStr(end_time - start_time)
            print('execution time so far: ', exec_time)

            optimizer.zero_grad()

            # Backward pass.
            loss.backward()

            # 1-step gradient descent.
            optimizer.step()

    end_time = time.time()
    exec_time = secondsToStr(end_time - start_time)
    print('execution time: ', exec_time)

    if FLAGS.save_path != None:
        # Save model
        ckpt_data = {}
        ckpt_data['epoch'] = shift_epoch + epoch
        ckpt_data['state_dict'] = net.state_dict()
        ckpt_data['optimizer'] = optimizer.state_dict()
        print(print_save)
        torch.save(ckpt_data, FLAGS.save_path + 'save_' + mode + '_' + FLAGS.model_choice + '_' + str(
            shift_epoch + epoch) + '_full_nomir.pth.tar')
        np.save(save_path + 'cat_scores_' + mode + '_' + FLAGS.model_choice + '_full_nomir.npy', cat_scores)
        np.save(save_path + 'trainloss_' + mode + '_' + FLAGS.model_choice + '_full_nomir.npy', np.array(trainloss))
        np.save(save_path + 'valloss_' + mode + '_' + FLAGS.model_choice + '_full_nomir.npy', np.array(valloss))


    return net, cat_scores, trainloss  # , valloss


def Acc(out, label, Print=0):
    # out and labels are tensors
    out, label = out.cpu(), label.cpu()
    out, label = np.argmax(out.detach().numpy(), axis=1), label.numpy()
    score = 100 * np.mean(out == label)
    print('out', out)
    print('label', label)
    print('')
    return score

