"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen2, VAEGen, UNet,NLayerDiscriminator
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, CrossEntropyLoss2d, norm, jaccard
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as functional
import os

import numpy as np

class MUNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None):

        super(MUNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        self.gen = AdaINGen2(hyperparameters['input_dim'], hyperparameters['gen'])  # Auto-encoder for domain a.
        self.dis = NLayerDiscriminator(hyperparameters['input_dim'])  # Discriminator for domain a.
        self.dis2 = NLayerDiscriminator(3*hyperparameters['input_dim'],n_layers=4)

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.weight_decay = hyperparameters['weight_decay']

        # Initiating and loader pretrained UNet.
        self.sup = UNet(input_channels=hyperparameters['input_dim'], num_classes=3).cuda()

        # Fix the noise used in sampling.
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis.parameters())
        dis2_params = list(self.dis2.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis2_opt = torch.optim.Adam([p for p in dis2_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.dis2_scheduler = get_scheduler(self.dis2_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))
        self.dis2.apply(weights_init('gaussian'))
        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256).cuda()
        self.one_hot_c = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64).cuda()

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_c[i, :, i, :, :].fill_(1)

        if resume_epoch != -1:

            self.resume(snapshot_dir, hyperparameters,resume_epoch)

    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        loss = CrossEntropyLoss2d(size_average=True).cuda()
        return loss(input, target)

    def forward(self, x_a, x_b):

        self.eval()

        x_a.volatile = True
        x_b.volatile = True

        s_a = Variable(self.s_a, volatile=True)
        s_b = Variable(self.s_b, volatile=True)

        c_a, s_a_fake = self.gen.encode(x_a)
        c_b, s_b_fake = self.gen.encode(x_b)

        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)

        self.train()

        return x_ab, x_ba

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True
##################################################################################
# Mainly adapted from https://github.com/hugo-oliveira/CoDAGANs ##################
##################################################################################
    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, ep,hyperparameters):

        self.gen_opt.zero_grad()

        # temp_open=hyperparameters['temp_open']
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)

        c_b_recon, s_a_recon = self.gen.encode(x_ba)
        c_a_recon, s_b_recon = self.gen.encode(x_ab)

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None
        # if temp_open==1:
        c_a=c_a.detach()
        c_b=c_b.detach()
        c_b_recon=c_b_recon.detach()
        c_a_recon=c_a_recon.detach()

        p_a = self.sup(c_a, use_a, True)
        p_a_recon = self.sup(c_a_recon, use_a, True)
        p_b = self.sup(c_b, use_a, True)
        p_b_recon = self.sup(c_b_recon, use_a, True)

        loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                          self.semi_criterion(p_a_recon, y_a[use_a, :, :])
        if (ep+1)>10:
            loss_gen_b = self.dis2.calc_gen_loss(p_b)+self.dis2.calc_gen_loss(p_b_recon)
        else:
            loss_gen_b = Variable(torch.tensor(0).cuda(), requires_grad=False) 
        self.loss_gen_total = None
        weight_temp=hyperparameters['weight_temp']
        if loss_semi_a is not None:
            self.loss_gen_total = hyperparameters['recon_x_w'] *loss_semi_a+weight_temp*loss_gen_b
            seg_loss = hyperparameters['recon_x_w'] *loss_semi_a
            seg_gen_loss = weight_temp*loss_gen_b
        if self.loss_gen_total is not None:
            self.loss_gen_total.backward()
            self.gen_opt.step()
        return seg_loss.item(),seg_gen_loss.item()

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        content, _ = self.gen.encode(x)

        # Forwarding on supervised model.
        y_pred = self.sup(content, only_prediction=True)

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        jacc,jacc_cup = jaccard(pred, y.cpu().squeeze(0).numpy())

        return jacc,jacc_cup, pred, content

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.

        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        # Decode (within domain).
        x_a_recon = self.gen.decode(c_a, s_a_prime)
        x_b_recon = self.gen.decode(c_b, s_b_prime)

        # Decode (cross domain).
        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)

        # Encode again.
        c_b_recon, s_a_recon = self.gen.encode(x_ba)
        c_a_recon, s_b_recon = self.gen.encode(x_ab)

        # Decode again (if needed).
        x_aba = self.gen.decode(c_a_recon, s_a_prime)
        x_bab = self.gen.decode(c_b_recon, s_b_prime)

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()
        return self.loss_gen_total.item()

    def compute_vgg_loss(self, vgg, img, target):

        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()
        
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        c_a, _ = self.gen.encode(x_a)
        c_b, _ = self.gen.encode(x_b)

        # Decode (cross domain).
        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)

        # D loss.
        self.loss_dis_a = self.dis.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()
        return self.loss_dis_total.item()


    def dis2_update(self, x_a, x_b, d_index_a, d_index_b,use_a,use_b, hyperparameters):
    
        self.dis2_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        c_a, s_a_prime = self.gen.encode(x_a)
        c_b, s_b_prime = self.gen.encode(x_b)

        # Decode (within domain).
        x_a_recon = self.gen.decode(c_a, s_a_prime)
        x_b_recon = self.gen.decode(c_b, s_b_prime)

        # Decode (cross domain).
        x_ba = self.gen.decode(c_b, s_a)
        x_ab = self.gen.decode(c_a, s_b)

        # Encode again.
        c_b_recon, s_a_recon = self.gen.encode(x_ba)
        c_a_recon, s_b_recon = self.gen.encode(x_ab)

        p_b = self.sup(c_b, use_a, True)
        p_b_recon = self.sup(c_b_recon, use_a, True)
        p_a = self.sup(c_a, use_a, True)
        p_a_recon = self.sup(c_a_recon, use_a, True)
        
        self.loss_dis2_b = self.dis2.calc_dis_loss(p_b.detach(), p_a.detach())+self.dis2.calc_dis_loss(p_b_recon.detach(), p_a_recon.detach())
        self.loss_dis2_total = hyperparameters['gan_w'] * self.loss_dis2_b 

        self.loss_dis2_total.backward()
        self.dis2_opt.step()
        return self.loss_dis2_total.item()

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.dis2_scheduler is not None:
            self.dis2_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters,resume_epoch):

        print("--> " + checkpoint_dir)

        # Load generator.
        last_model_name = get_model_list(checkpoint_dir, "gen", resume_epoch)
        # print('\n',last_model_name)
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)
        epochs = int(last_model_name[-11:-3])

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, "sup", resume_epoch)
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load discriminator.
        # last_model_name = get_model_list(checkpoint_dir, "dis", resume_epoch)
        # state_dict = torch.load(last_model_name)
        # self.dis.load_state_dict(state_dict)

        # # Load discriminator2.
        # last_model_name = get_model_list(checkpoint_dir, "dis2", resume_epoch)
        # state_dict = torch.load(last_model_name)
        # self.dis2.load_state_dict(state_dict)
        
        # # Load optimizers.
        # last_model_name = get_model_list(checkpoint_dir, "opt", resume_epoch)
        # state_dict = torch.load(last_model_name)
        # self.dis_opt.load_state_dict(state_dict['dis'])
        # self.dis2_opt.load_state_dict(state_dict['dis2'])
        # self.gen_opt.load_state_dict(state_dict['gen'])

        # for state in self.dis_opt.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()

        # for state in self.dis2_opt.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
                    
        # for state in self.gen_opt.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()

        # # Reinitilize schedulers.
        # self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, epochs)
        # self.dis2_scheduler = get_scheduler(self.dis2_opt, hyperparameters, epochs)
        # self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, epochs)

        # print('Resume from epoch %d' % epochs)
        # return epochs

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        dis2_name = os.path.join(snapshot_dir, 'dis2_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.dis2.state_dict(), dis2_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict(),'dis2': self.dis2_opt.state_dict()}, opt_name)
