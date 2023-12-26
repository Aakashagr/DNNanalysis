# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:47:35 2021

@author: Aakash
"""
import time, gc, os, sys
import torch
import argparse
import torch.nn as nn
from torchvision import datasets, models, transforms
from time import strftime, localtime
from datetime import timedelta
import numpy as np
from torch.utils import data

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--wrd_path', default= 'wordsets_FR',
                    help='path to word folder that contains train and val folders')
parser.add_argument('--save_path', default='save/',
                    help='path for saving ')
parser.add_argument('--img_classes', default=1000,
                    help='number of image classes')
parser.add_argument('--wrd_classes', default=1000,
                    help='number of word classes')
parser.add_argument('--num_train_items', default=1300,
                    help='number of training items in each category')
parser.add_argument('--num_val_items', default=50,
                    help='number of validation items in each category')
parser.add_argument('--num_workers', default=10,
                    help='number of workers to load batches in parallel')
parser.add_argument('--max_epochs_lit', default=30, type=int,
                    help='number of epochs to run as literate - training on images and words')
parser.add_argument('--batch_size', default=100, type=int,
                    help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=.001, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_schedule', default='StepLR')
parser.add_argument('--step_size', default=20, type=int,
                    help='after how many epoch learning rate should be decreased 10x')
parser.add_argument('--momentum', default=.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay ')

FLAGS, _ = parser.parse_known_args()
import alexnet_def
import labels_word

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def train(save_path=FLAGS.save_path):
    start_time = time.time()

    global net
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


    print('building literate model')
    # Datasets and Generators
    transform_img = {
	    'train': transforms.Compose([
	        transforms.RandomResizedCrop(224),
	        transforms.RandomHorizontalFlip(),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	    'val': transforms.Compose([
	        transforms.Resize(256),
	        transforms.CenterCrop(224),
	        transforms.ToTensor(),
	        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ]),
	}

    ##############################################
    # # Initialize Imagenet Datasets for Jean Zay
    root_imagenet = os.path.join(os.environ["DSDIR"], "imagenet", "RawImages")
    print(root_imagenet)
    print(sys.version_info)
    train_imgset = datasets.ImageNet(root=root_imagenet, split='train', download=False, transform = transform_img['train'])
    val_imgset = datasets.ImageNet(root=root_imagenet, split='val', download=False, transform = transform_img['val'])
    #############################################

    transform_words = transforms.Compose(
         [transforms.RandomResizedCrop(224, scale = (0.9,1), ratio= (1,1)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    print('loading datasets')
    train_wrdset = datasets.ImageFolder(root= os.path.join(FLAGS.wrd_path, 'train'), transform=transform_words, target_transform = labels_word.labels_word)
    val_wrdset = datasets.ImageFolder(root= os.path.join(FLAGS.wrd_path, 'val'), transform=transform_words, target_transform = labels_word.labels_word)

    # Concatenating word and image dataset
    train_set = torch.utils.data.ConcatDataset((train_imgset, train_wrdset))
    val_set = torch.utils.data.ConcatDataset((val_imgset, val_wrdset))

    # Loading data
    training_gen = data.DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)
    validation_gen = data.DataLoader(val_set, batch_size=FLAGS.num_val_items, shuffle=False, num_workers=FLAGS.num_workers)

    # variables, labels, prints, and titles for plots
    print('loading variables')
    classes = FLAGS.img_classes + FLAGS.wrd_classes
    max_epochs = FLAGS.max_epochs_lit
    cat_scores = np.zeros((FLAGS.max_epochs_lit, classes))
    print('np.shape(cat_scores)', np.shape(cat_scores))

    # Model
    print('loading Alexnet model')
    net = alexnet_def.AlexNet()
    model_params = torch.load('alexnet_trained.pth')
    net.load_state_dict(model_params)
    net.classifier._modules['6'] = torch.nn.Linear(4096, 2000)
    net.classifier[6].bias.data[:1000] = model_params['classifier.6.bias']
    net.classifier[6].weight.data[:1000,:] = model_params['classifier.6.weight']
    trainloss, valloss = [], []

    # use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    # transfer model to device
    net.to(device)


    exec_time = secondsToStr(time.time() - start_time)
    print('execution time so far: ', exec_time)


    criterion = nn.CrossEntropyLoss()     # Build loss function, model and optimizer.
    optimizer = torch.optim.SGD(net.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.weight_decay)  # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)


    """
    train
    """

    # Loop over epochs
    for epoch in range(max_epochs):

        gc.collect()
        # Training
        print('epoch', epoch)

        # Validation
        with torch.set_grad_enabled(False):
            cat_index = 0; net.eval()
            for inputs, labels in validation_gen:
                print('cat_index', cat_index)

                # Transfer to GPU
                inputs, labels = inputs.to(device), labels.to(device)

                # Model computations
                pred_val = net(inputs)
                scores = Acc(pred_val, labels)

                print('category accuracy scores', scores)
                cat_scores[epoch, cat_index] = scores
                print('cat_scores[epoch, cat_index]', cat_scores[epoch, cat_index])
                exec_time = secondsToStr(time.time() - start_time)
                print('execution time so far: ', exec_time)
                cat_index += 1

                # Compute loss.
                loss_val = criterion(pred_val, labels)
                valloss += [loss_val.item()]

        with torch.set_grad_enabled(True):
            net.train()
            for inputs, labels in training_gen:
	            # Transfer to GPU
	            inputs, labels = inputs.to(device), labels.to(device)
	            torch.cuda.empty_cache()

	            # Forward pass.
	            pred_train = net(inputs)

	            # Compute loss.
	            loss = criterion(pred_train, labels)
	            trainloss += [loss.item()]
	            print('epoch :', epoch, ', loss: ', loss.item())
	            end_time = time.time()
	            exec_time = secondsToStr(end_time - start_time)
	            print('execution time so far: ', exec_time)

	            optimizer.zero_grad()
	            loss.backward()   # Backward pass
	            optimizer.step()  # 1-step gradient descent.
# 	            scheduler.step()



        if FLAGS.save_path is not None:
            # Save model
            ckpt_data = {}
            ckpt_data['epoch'] = epoch
            ckpt_data['state_dict'] = net.state_dict()
            ckpt_data['optimizer'] = optimizer.state_dict()
            print('Saving literate Alexnet')
            torch.save(ckpt_data, FLAGS.save_path + 'save_Alexnet_lit_' + str(epoch) + '.pth.tar')
#             torch.save(net.module.state_dict(), FLAGS.save_path + 'save_Alexnet_lit_' + str(epoch) + '.pth.tar')
            np.save(save_path + 'cat_scores_Alexnet_lit.npy', cat_scores)
            np.save(save_path + 'trainloss_Alexnet_lit.npy', np.array(trainloss))
            np.save(save_path + 'valloss_Alexnet_lit.npy', np.array(valloss))


    end_time = time.time()
    exec_time = secondsToStr(end_time - start_time)
    print('execution time: ', exec_time)

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
