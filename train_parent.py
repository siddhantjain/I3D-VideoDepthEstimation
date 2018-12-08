# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from util import visualize as viz
from dataloaders import kitti_eigen as db
from dataloaders import custom_transforms as tr
import networks.i3d_osvos as vos
from layers.osvos_layers import class_balanced_cross_entropy_loss
from mypath import Path
from logger import Logger
from dataloaders import helpers


# Select which GPU, -1 if CPU
gpu_id = 0
print('Using GPU: {} '.format(gpu_id))

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 4,  # Number of Images in each mini-batch
}

# # Setting other parameters
resume_epoch = 0  # Default is 0, change if want to resume
nEpochs = 500  # Number of epochs for training (500.000/2079)
useTest = True  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 5  # Run on test set every nTestInterval epochs
db_root_dir = Path.db_root_dir()
vis_net = 0  # Visualize the network?
snapshot = 40  # Store a model every snapshot epochs
nAveGrad = 1
train_rgb = True
save_dir = Path.save_root_dir()
if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))

# Network definition
#siddhanj: here number of classes do not matter, as that just helps choose what kind of i3d video we want to use
modelName = 'parent'

#num_classes is irrelevant here but it corresponds to the number of filters of last layer convolutions before upsampling blocks
netRGB = vos.I3D(num_classes=400, modality='rgb')

#Sowmya: Depth Estimation: Uncomment this if you want to load weights for pretraining
#netRGB.load_state_dict(torch.load('models/parent_epoch-199.pth'),False)

# Logging into Tensorboard
tboardLogger = Logger('../logs/tensorboardLogs', 'train_parent')

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    netRGB.cuda()


# Sowmya: Use Adam brah!
# Use the following optimizer
lr = 1e-4
wd = 0.0002
_momentum = 0.9
optimizer = optim.SGD(netRGB.parameters(), lr, momentum=_momentum,
                                            weight_decay=wd)


# Preparation of the data loaders
# Define augmentation transformations as a composition

#composed_transforms = transforms.Compose([tr.VideoLucidDream(), tr.ToTensor()])

#composed_transforms = transforms.Compose([tr.ToTensor()])

# Training dataset and its iterator
db_train = db.KITTI(mode='train',
             input_res=[512,348],
             db_root_dir='/home/siddhanj/16822/monodepth-data',
             central_frame_list='train_file_list.txt',
             path_to_depth_npy='/home/siddhanj/16822/code/monodepth/models/disparities_pp.npy',
             path_to_valid_files='allfiles.txt',
             seq_length=8)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)

running_loss_tr = [0] * 5
running_loss_ts = [0] * 5
loss_tr = []
loss_ts = []
aveGrad = 0


def getDepthFrom2DArray(inArray):
    inArray_ = (inArray - np.min(inArray))/(np.max(inArray) - np.min(inArray))
    cm = plt.get_cmap('jet')
    retValue = cm(inArray_)
    return retValue

print("Training Network")
# Main Training and Testing Loop
start_step = 0
for epoch in range(resume_epoch, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)

        inputs = torch.transpose(inputs, 1, 2)
        gts = torch.transpose(gts, 1, 2)
        if gpu_id >= 0:
            inputs, gts = inputs.cuda(), gts.cuda()


        outputs = netRGB.forward(inputs)

        if ii == len(trainloader) - 1:
            images_list = []
            number_frames = inputs.shape[2]
            logging_frames = np.arange(number_frames)
            inputs_ = torch.transpose(inputs, 1, 2)
            outputs_ = torch.transpose(outputs[-1], 1, 2)
            if gpu_id >= 0:
                all_inputs = inputs_.data.cpu().numpy()[0,logging_frames,:,:,:]
                all_outputs = outputs_.data.cpu().numpy()[0,logging_frames,:,:,:]
            else:
                all_inputs = inputs_.data.numpy()[0,logging_frames,:,:,:]
                all_outputs = outputs_.data.numpy()[0,logging_frames,:,:,:]
            for imageIndex in range(number_frames):
                inputImage = all_inputs[imageIndex, :, :, :]
                inputImage = np.transpose(inputImage, (1, 2, 0))
                inputImage = inputImage[:,:,::-1]
                images_list.append(inputImage)
                depth = all_outputs[imageIndex, 0, :, :]
                invDepthMap = getDepthFrom2DArray(depth)
                images_list.append(invDepthMap)
            tboardLogger.image_summary('image_{}'.format(epoch), images_list, epoch)

        # Compute the losses, side outputs and fuse

        losses = [0] * len(outputs)
        for i in range(0, len(outputs)):
            losses[i] = vos.compute_loss(outputs[i].squeeze(), gts[i].squeeze())
            running_loss_tr[i] += losses[i].data[0]
        loss = (1 - epoch / nEpochs)*sum(losses[:-1]) + losses[-1]

        '''
        # Print stuff
        if ii % num_img_tr == num_img_tr - 1:
            running_loss_tr = [x / num_img_tr for x in running_loss_tr]
            loss_tr.append(running_loss_tr[-1])
            print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
            for l in range(0, len(running_loss_tr)):
                print('Loss %d: %f' % (l, running_loss_tr[l]))
                running_loss_tr[l] = 0

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time))
        '''
        print('[Training Information: Epoch: %d, loss: %f]' % (epoch, loss))
        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1
        tboardLogger.scalar_summary("training_loss",loss.data[0],start_step+ii)

        optimizer.step()
        optimizer.zero_grad()
        aveGrad = 0

    start_step = start_step + len(trainloader)
    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        torch.save(netRGB.state_dict(), os.path.join(save_dir, modelName + '_epoch-' + str(epoch) + '.pth'))


    '''
    Siddhant: Not doing this right now. Will come back to this, once we have test code ready?
    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        for ii, sample_batched in enumerate(testloader):
            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward pass of the mini-batch
            inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            outputs = net.forward(inputs)

            # Compute the losses, side outputs and fuse
            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_ts[i] += losses[i].data[0]
            loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

            # Print stuff
            if ii % num_img_ts == num_img_ts - 1:
                running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                loss_ts.append(running_loss_ts[-1])

                print('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                writer.add_scalar('data/test_loss_epoch', running_loss_ts[-1], epoch)
                for l in range(0, len(running_loss_ts)):
                    print('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    running_loss_ts[l] = 0
    '''

