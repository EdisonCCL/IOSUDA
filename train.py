"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, norm
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # Will be 3.x series.
    pass
import os
import sys
import math
import shutil
import numpy as np

from skimage import io
from tensorboardX import SummaryWriter
import socket
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="Outputs path.")
parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--snapshot_dir', type=str, default='./outputs/config/checkpoints')
parser.add_argument('--n_datasets',type=int,default=4)
parser.add_argument('--data_root',type=str,default='./datasets/retinal_data/')
parser.add_argument('--snapshot_save_iter', type=int, default=10)
parser.add_argument('--sample_C',type=float,default=0.0)
parser.add_argument('--sample_D',type=float,default=0.0)
parser.add_argument('--sample_A',type=float,default=0.0)
parser.add_argument('--sample_B',type=float,default=1.0)
parser.add_argument('--trim',type=int,default=0)
parser.add_argument('--batch_size',type=int,default=2)
parser.add_argument('--transform_A',type=int,default=2)
parser.add_argument('--transform_B',type=int,default=2)
parser.add_argument('--transform_C',type=int,default=2)
parser.add_argument('--transform_D',type=int,default=2)
parser.add_argument('--dataset_letters',type=str,default="['B','A']")
parser.add_argument('--test',type=int,default=1)
parser.add_argument('--weight_temp',type=float,default=1)
opts = parser.parse_args()
#print(opts)
cudnn.benchmark = True

# Load experiment setting.
config = get_config(opts.config)
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path
config['n_datasets']=opts.n_datasets
config['data_root']=opts.data_root
config['snapshot_dir']=opts.snapshot_dir
config['snapshot_save_iter']=opts.snapshot_save_iter
config['sample_C']=opts.sample_C
config['trim']=opts.trim
config['sample_B']=opts.sample_B
config['sample_D']=opts.sample_D
config['sample_A']=opts.sample_A
config['batch_size']=opts.batch_size
config['transform_A']=opts.transform_A
config['transform_B']=opts.transform_B
config['transform_C']=opts.transform_C
config['transform_D']=opts.transform_D
config['weight_temp']=opts.weight_temp

# Setup model and data loader.
trainer = MUNIT_Trainer(config, resume_epoch=opts.resume, snapshot_dir=opts.snapshot_dir)
trainer.cuda()

dataset_letters = eval(opts.dataset_letters)
samples = list()
dataset_probs = list()
augmentation = list()
for i in range(config['n_datasets']):
    samples.append(config['sample_' + dataset_letters[i]])
    augmentation.append(config['transform_' + dataset_letters[i]])

train_loader_list, test_loader_list = get_all_data_loaders(config, config['n_datasets'], samples, augmentation, config['trim'],opts.dataset_letters)

loader_sizes = list()

for l in train_loader_list:

    loader_sizes.append(len(l))

loader_sizes = np.asarray(loader_sizes)
n_batches = loader_sizes.min()

# Setup logger and output folders.
model_name = os.path.splitext(os.path.basename(opts.config))[0]
# model_name=config['model_name']
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # Copy config file to output folder.

# Start training.
epochs = config['max_iter']
log_dir = os.path.join(opts.output_path, 'tensorboard',datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=log_dir)

for ep in range(max(opts.resume, 0), epochs):

    print('Start of epoch ' + str(ep + 1) + '...')

    trainer.update_learning_rate()
    seg_loss=0.0
    seg_gen_loss=0.0
    gen_loss=0.0
    dis_loss=0.0
    dis2_loss=0.0
    print('    Training...')
    for it, data in enumerate(zip(*train_loader_list)):
        # print(data)
        images_list = list()
        labels_list = list()
        use_list = list()

        for i in range(config['n_datasets']):
            images = data[i][0]
            labels = data[i][1]
            use = data[i][2].to(dtype=torch.uint8)
            images_list.append(images)
            labels_list.append(labels)
            use_list.append(use)

        print('        Ep: ' + str(ep + 1) + ', it: ' + str(it + 1) + '/' + str(n_batches))

        index_1 = 0
        index_2 = 1

        images_1 = images_list[index_1]
        images_2 = images_list[index_2]
        labels_1 = labels_list[index_1]
        labels_2 = labels_list[index_2]

        use_1 = use_list[index_1]
        use_2 = use_list[index_2]
        #print(use_1,use_2)
        images_1, images_2 = Variable(images_1.cuda()), Variable(images_2.cuda())

        # Main training code.
        if (ep + 1) <= int(0.75 * epochs):

            # If in Full Training mode.
            trainer.set_sup_trainable(True)
            trainer.set_gen_trainable(True)

            dis_loss+=trainer.dis_update(images_1, images_2, index_1, index_2, config)
            gen_loss+=trainer.gen_update(images_1, images_2, index_1, index_2, config)

        else:
            # If in Supervision Tuning mode.
            trainer.set_sup_trainable(True)
            trainer.set_gen_trainable(False)

        labels_1 = labels_1.to(dtype=torch.long)
        labels_1 = Variable(labels_1.cuda(), requires_grad=False)       
        labels_2 = labels_2.to(dtype=torch.long)
        labels_2 = Variable(labels_2.cuda(), requires_grad=False)

        if (ep+1)<=10:
            temp_loss=trainer.sup_update(images_1, images_2, labels_1, labels_2, index_1, index_2, use_1, use_2,ep, config)   
            seg_loss+=temp_loss[0]
            seg_gen_loss+=temp_loss[1] 
        else:
            temp_loss=trainer.sup_update(images_1, images_2, labels_1, labels_2, index_1, index_2, use_1, use_2,ep, config)   
            seg_loss+=temp_loss[0]
            seg_gen_loss+=temp_loss[1] 
            dis2_loss+=trainer.dis2_update(images_1,images_2,index_1, index_2, use_1, use_2, config)
    gen_loss=gen_loss/(it+1)
    seg_loss=seg_loss/(it+1)
    seg_gen_loss=seg_gen_loss/(it+1)
    dis_loss=dis_loss/(it+1)
    dis2_loss=dis2_loss/(it+1)

    writer.add_scalar('train_seg/seg_loss', seg_loss, ep+1)
    writer.add_scalar('train_seg2/seg_gen_loss', seg_gen_loss, ep+1)
    writer.add_scalar('train_dis/dis_loss', dis_loss, ep+1)
    writer.add_scalar('train_dis2/dis2_loss', dis2_loss, ep+1)
    writer.add_scalar('train_gen/gen_loss', gen_loss, ep+1)
    if (ep + 1) % config['snapshot_save_iter'] == 0:
        trainer.save(checkpoint_directory, (ep + 1))
        if opts.test==1:
            print('    Testing ...')
    
            jacc_list = list()
            jacc_cup_list = list()
            for it, data in enumerate(test_loader_list[1]):
                images = data[0]
                labels = data[1]
                use = data[2]
                path = data[3]
                images = Variable(images.cuda())
                labels = labels.to(dtype=torch.long)
                labels = Variable(labels.cuda(), requires_grad=False)
                jacc,jacc_cup, pred, iso = trainer.sup_forward(images, labels, 0, config)
                jacc_list.append(jacc)
                jacc_cup_list.append(jacc_cup)         
            jaccard = np.asarray(jacc_list)
            jaccard_cup = np.asarray(jacc_cup_list)
            writer.add_scalar('val_data/val_CUP_dice', 100*jaccard_cup.mean(), ep+1)
            writer.add_scalar('val_data/val_DISC_dice', 100*jaccard.mean(), ep+1)
                # print('        Test ' + dataset_letters[i] + ' Jaccard epoch ' + str(ep + 1) + ': ' + str(100 * jaccard.mean()) + ' +/- ' + str(100 * jaccard.std()))
