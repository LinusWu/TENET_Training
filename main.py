# -*- coding: utf-8 -*-
from config import *
import Data.get_data as _Data
import Models.get_model as _Models
from tqdm import *
import os
from utils import TopK, AverageMeter
import numpy as np
import torch
import TENET.Tenet as _TENET
import third_party.adv_free_AT.free_AT as AT
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_


def train(config, model, train_loader, test_loader, loss_fun, optimizer, lr_scheduler):
    global_ite = train_top1 = train_top5 = val_top1 = val_top5 = 0
    loss_value = loss_tenet_value = loss_orth_value  = 0
    
    Tenet = None
    ACC_list = []
    for _ in range(config.save_model_num):
        ACC_list.append(config.save_acc_thres * 100)

    model.cuda()
    if config.mode == 'RL':
        Tenet = _TENET.Tenet(model=model, group_numbers=config.num_clusters, feature_id = 3)

    for e in range(config.epoch):
        train_dataiter = iter(train_loader)
        t = tqdm(train_dataiter)
        t.set_description("TENET Epoch [{}/{}]".format(e+1,config.epoch))
        # test_times = 0
        # resnet.cuda()
        for i, (images, labels) in enumerate(t):
            global_ite = global_ite + 1

            # reset model's mode
            model.train()

            # data and labels
            images = images.float().cuda()
            labels = labels.long().cuda()

            # inference
            out = model(images)
            loss = loss_fun(out,labels)

            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get loss and acc
            loss_value = loss.cpu().detach().data.numpy()
            train_top1, train_top5 = TopK(out.data, labels.data, topk=(1, 5))


            # test stage and save TOP-K models
            if global_ite % config.test_times == 0:
                val_top1, val_top5 = test(config, model, test_loader, ACC_list)
                
            
            # TENET stage
            if Tenet is not None and global_ite >= config.TENET_Switch_Number and global_ite % config.thinking_alpha == 0 and train_top1 > 0:
                # select positive samples
                out_index = out.argmax(dim=1)
                index = torch.where(out_index==labels)
                _images = torch.index_select(images, 0, index[0])
                _labels = torch.index_select(labels, 0, index[0])

                # TENET begins
                output, loss_orth = Tenet(_images, _labels)
                loss_tenet = config.loss_tenet_lambda * loss_fun(output,_labels) + config.orth_loss_lambda * loss_orth

                # backward and update
                optimizer.zero_grad()
                loss_tenet.backward()
                optimizer.step()

                # get loss and acc
                loss_tenet_value = loss_tenet.cpu().detach().data.numpy()
                loss_orth_value = loss_orth.cpu().detach().data.numpy()

            t.set_postfix_str('train-ACCTOP1/5={:^7.3f}/{:^7.3f}, val_mACCTOP1/5:{:^7.3f}/{:^7.3f}, Tenet_Loss:{:^7.3f}, Orth_Loss:{:^7.3f}, train_Loss:{:^7.3f}'
                .format(train_top1, train_top5, val_top1, val_top5 ,loss_tenet_value ,loss_orth_value ,loss_value))
            t.update()
        lr_scheduler.step()

def train_cifar(config, model, train_loader, test_loader, loss_fun, optimizer, lr_scheduler):
    print("train cifar")
    global_ite = train_top1 = train_top5 = val_top1 = val_top5 = 0
    loss_value = loss_tenet_value = loss_orth_value  = 0
    cifar_10_epoch = -1
    cifar_100_epoch = -1
    Tenet = None
    ACC_list = []
    for _ in range(config.save_model_num):
        ACC_list.append(config.save_acc_thres * 100)

    model.cuda()
    if config.mode == 'RL':
        Tenet = _TENET.Tenet(model=model, group_numbers=config.num_clusters, feature_id = 3)

    for e in range(config.epoch):
        train_dataiter = iter(train_loader)
        t = tqdm(train_dataiter)
        t.set_description("TENET Epoch [{}/{}]".format(e+1,config.epoch))
        # test_times = 0
        # resnet.cuda()
        for i, (images, labels) in enumerate(t):
            global_ite = global_ite + 1
            if (e > cifar_10_epoch and config.dataset.lower() == 'cifar10') or (e > cifar_100_epoch and config.dataset.lower() == 'cifar100') :

                # reset model's mode
                model.train()

                # data and labels
                images = images.float().cuda()
                labels = labels.long().cuda()

                # inference
                out = model(images)
                loss = loss_fun(out,labels)

                # backward and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # get loss and acc
                loss_value = loss.cpu().detach().data.numpy()
                train_top1, train_top5 = TopK(out.data, labels.data, topk=(1, 5))


                # test stage and save TOP-K models
                if global_ite % config.test_times == 0:
                    val_top1, val_top5 = test(config, model, test_loader, ACC_list)
                    
                
                # TENET stage
                if Tenet is not None and global_ite >= config.TENET_Switch_Number and global_ite % config.thinking_alpha == 0 and train_top1 > 0:
                    # select positive samples
                    out_index = out.argmax(dim=1)
                    index = torch.where(out_index==labels)
                    _images = torch.index_select(images, 0, index[0])
                    _labels = torch.index_select(labels, 0, index[0])

                    # TENET begins
                    output, loss_orth = Tenet(_images, _labels)
                    loss_tenet = config.loss_tenet_lambda * loss_fun(output,_labels) + config.orth_loss_lambda * loss_orth

                    # backward and update
                    optimizer.zero_grad()
                    loss_tenet.backward()
                    optimizer.step()

                    # get loss and acc
                    loss_tenet_value = loss_tenet.cpu().detach().data.numpy()
                    loss_orth_value = loss_orth.cpu().detach().data.numpy()

                t.set_postfix_str('train-ACCTOP1/5={:^7.3f}/{:^7.3f}, val_mACCTOP1/5:{:^7.3f}/{:^7.3f}, Tenet_Loss:{:^7.3f}, Orth_Loss:{:^7.3f}, train_Loss:{:^7.3f}'
                    .format(train_top1, train_top5, val_top1, val_top5 ,loss_tenet_value ,loss_orth_value ,loss_value))
                t.update()
            lr_scheduler.step()


def train_adv_cifar(config, model, train_loader, test_loader, loss_fun, optimizer, lr_scheduler):
    global_ite = train_top1 = train_top5 = val_top1 = val_top5 = 0
    loss_value = loss_tenet_value = loss_orth_value  = 0
    
    Tenet = None
    ACC_list = []
    for _ in range(config.save_model_num):
        ACC_list.append(config.save_acc_thres * 100)

    model.cuda()
    if config.mode == 'RL':
        Tenet = _TENET.Tenet(model=model, group_numbers=config.num_clusters, feature_id = 3)

    global_noise_data = torch.zeros([config.train_batch_size, 3, config.crop_size, config.crop_size]).cuda()

    mean = torch.Tensor(np.array([0.5,0.5,0.5])[:, np.newaxis, np.newaxis])
    mean = mean.expand(3,config.crop_size, config.crop_size).cuda()

    std = torch.Tensor(np.array([0.5,0.5,0.5])[:, np.newaxis, np.newaxis])
    std = std.expand(3, config.crop_size, config.crop_size).cuda()

    for e in range(config.epoch):
        train_dataiter = iter(train_loader)
        t = tqdm(train_dataiter)
        t.set_description("TENET Epoch [{}/{}]".format(e+1,config.epoch))
        # test_times = 0
        # resnet.cuda()
        for i, (images, labels) in enumerate(t):
            global_ite = global_ite + 1

            # # reset model's mode
            model.train()

            # # data and labels
            images = images.float().cuda(non_blocking=True)
            labels = labels.long().cuda(non_blocking=True)

            # test stage and save TOP-K models
            if global_ite % config.test_times == 0:
                val_top1, val_top5 = test(config, model, test_loader, ACC_list)
                
            
            # adv training
            loss_AT = 0
            loss_tenet_value = 0
            loss_orth_value = 0
            for j in range(config.ADV_n_repeats):
                # produce adv data
                noise_batch = Variable(global_noise_data[0:labels.size(0)], requires_grad=True).cuda()
                
                in1 = images + noise_batch

                in1.clamp_(0.0, 1.0)
                in1.sub_(mean).div_(std)
                output = model(in1)
                # forward and backward
                loss = loss_fun(output, labels)
                loss_AT += loss.cpu().detach().data.numpy()
                

                # update model param
                # compute gradient and do SGD step
                optimizer.zero_grad()
                if Tenet is not None:
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                # Update the noise for the next iteration
                pert = AT.fgsm(noise_batch.grad, config.ADV_fgsm_step)

                global_noise_data[0:labels.size(0)] += pert.data
                global_noise_data.clamp_(-config.ADV_clip_eps, config.ADV_clip_eps)
                optimizer.step()

                train_top1, train_top5 = TopK(output.data, labels.data, topk=(1, 5))

                # use Tenet strategy (optional)
                # TENET stage
                if Tenet is not None and global_ite >= config.TENET_Switch_Number:
                    # select positive samples
                    optimizer.zero_grad()
                    out_index = output.argmax(dim=1)
                    index = torch.where(out_index==labels)
                    _images = torch.index_select(in1, 0, index[0]).clone()
                    _labels = torch.index_select(labels, 0, index[0]).clone()
                    
                    if _images.size(0) > 0:
                        # TENET begins
                        output, loss_orth = Tenet(_images, _labels)
                        loss_tenet = config.loss_tenet_lambda * loss_fun(output,_labels) + config.orth_loss_lambda * loss_orth

                        # backward and update
                        optimizer.zero_grad()
                        loss_tenet.backward()
                        optimizer.step()

                        # get loss and acc
                        loss_tenet_value += loss_tenet.cpu().detach().data.numpy()
                        loss_orth_value += loss_orth.cpu().detach().data.numpy()

            loss_tenet_value = loss_tenet_value / config.ADV_n_repeats
            loss_orth_value = loss_orth_value / config.ADV_n_repeats
            loss_AT = loss_AT / config.ADV_n_repeats
            t.set_postfix_str('lr={:^7.3f}, train-ACCTOP1/5={:^7.3f}/{:^7.3f}, val_mACCTOP1/5:{:^7.3f}/{:^7.3f}, Tenet_Loss:{:^7.3f}, Orth_Loss:{:^7.3f}, train_Loss:{:^7.3f}, AT_loss{:^7.3f}'
                .format(lr_scheduler.get_last_lr()[0], train_top1, train_top5, val_top1, val_top5 ,loss_tenet_value ,loss_orth_value ,loss_value,loss_AT))
            t.update()
            lr_scheduler.step()



def test(config, model, test_loader, ACC_list):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for _, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            prec1, prec5 = TopK(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    if top1.avg >= min(ACC_list):
        if os.path.exists(config.save_model_path) is False:
            os.makedirs(config.save_model_path)

        ACC_list[ACC_list.index(min(ACC_list))] = top1.avg
        ACC_list = sorted(ACC_list, reverse=True)

        path_ = os.path.join(config.save_model_path, config.mode +'_TOP_'+str(ACC_list.index(top1.avg)+1) +'_Net.pth')
        torch.save(model.state_dict(),path_)
        print("Save model! Path:{} mACCTOP1/TOP5: {:4f}/{:4f}".format(path_, top1.avg, top5.avg))
    model.train()
    return top1.avg, top5.avg


def run(config):
    # data 
    train_loader, test_loader   = _Data.get_data(config)
    # model & loss
    model, loss_fun             = _Models.get_model_loss(config)
    # optimizer & scheduler
    optimizer, lr_scheduler     = _Models.get_opti_scheduler(config, model, train_loader)
    # training & online testing
    if 'cifar' in  config.dataset.lower() and config.AT:
        train_adv_cifar(config, model, train_loader, test_loader, loss_fun, optimizer, lr_scheduler)

    elif 'cifar' in  config.dataset.lower() and config.AT == False:
        train_cifar(config, model, train_loader, test_loader, loss_fun, optimizer, lr_scheduler)
    else:
        train(config, model, train_loader, test_loader, loss_fun, optimizer, lr_scheduler)


if __name__ == '__main__':
    config = set_config()
    run(config)



    
    
