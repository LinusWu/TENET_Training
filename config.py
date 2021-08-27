# -*- coding: utf-8 -*-
import time
import os
import argparse


class DefaultConfig(object):
    def __init__(self):
        self.root = './'
        self.pretrain_model = 'ResNet18'
        self.pretrain_path = './'
        self.test_dir_for_code = './test/'
        self.save_model_path = './Model/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        self.save_model_path = self.save_model_path + self.current_time
        self.augmentation_switch=True
        self.is_ten_crop = False
        self.epoch = 50
        self.train_batch_size = 256
        self.train_batch_num = 1 
        self.test_batch_num = 256 
        self.test_times = 2
        self.network_name = 'Resnet18'
        self.loss = 'BCEloss'
        self.classifier_optimizer = 'Adam'

        self.classifier_learning_rate = 1e-4

        #network is trained using TENET strategy per thinking_alpha training times
        self.thinking_alpha = 1 
        self.save_mAP_thres = 0.76 
        self.save_model_num = 5
        self.orth_loss_lambda = 1e-1
        self.lr_decay_step = 6
        self.lr_decay_gamma = 0.8
        self.num_clusters = 6
        self.loss_tenet_lambda = 1e-1

class ConfigForCIFAR(DefaultConfig):
    def __init__(self, num_classes=10):
        DefaultConfig.__init__(self)

        self.model = 'resnext29'
        self.pretrain = False
        self.num_classes = num_classes
        self.save_model_path = './Model/CIFAR'+str(self.num_classes)+'/'+self.model+'/'
        self.current_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        self.save_model_path = self.save_model_path + self.current_time
        
        self.augmentation_switch = True

        self.epoch = 200
        self.train_batch_size = 256
        self.test_batch_size = 100
        self.test_times = 25

        self.loss = 'CrossEntropy'
        self.classifier_optimizer = 'SGD'
        self.classifier_learning_rate = 1e-1
        self.thinking_alpha = 3
        self.TENET_Switch_Number = 500
        self.aug_mix_severity = 3
        self.save_acc_thres = 0.8
        self.save_model_num = 5
        if self.num_classes == 100:
            self.save_acc_thres = self.save_acc_thres - 0.2

        self.tensorboard_log_dir = './visualization/CIFAR'+str(self.num_classes)+'/tensorboard_'
        self.sample_image_iter = 5
        self.lr_decay_step = 40
        self.lr_decay_gamma = 0.5
        self.num_clusters = 6
        self.crop_size = 32
        self.ADV_n_repeats = 4
        self.ADV_fgsm_step = 4.0 / 255
        self.ADV_clip_eps =  4.0 / 255
        self.AT = False
        if self.AT == True:
            self.epoch = 50
        self.loss_tenet_lambda = 1e-1
        self.orth_loss_lambda = 1e-3 


def init(args):
    config = None
    if args.dataset.lower() == 'cifar10':
        config = ConfigForCIFAR(10)
    elif args.dataset.lower() == 'cifar100':
        config = ConfigForCIFAR(100)

    # append args info into the config
    for k, v in vars(args).items():
        setattr(config, k, v)

    return config

def set_config():
    parser = argparse.ArgumentParser(description='TENET Training')
    parser.add_argument("--mode", type=str, default='RL') 
    parser.add_argument("--dataset", type=str, default='CIFAR10')
    parser.add_argument("--resume_model_path", type=str, default='')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument("--group_num", type=int, default=6)
    parser.add_argument("--augmentation_switch", type=bool, default=True)
    parser.add_argument("--data_root", type=str, default='CIFAR10',help='the dir of dataset')
    args = parser.parse_args()

    # select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # init
    config = init(args)
    return config

