import sys
sys.path.append('./Models/')
from resnext import ResneXt as Resnext29
from resnet import ResNet,BasicBlock,Bottleneck
from torchvision_models import load_pretrained, pretrained_settings
import torch
import torch.nn.functional as F
import numpy as np

def Resnext29_init(config):
    if config.dataset.lower() == 'cifar10':
        model = Resnext29(num_classes=10) 
    
    if config.dataset.lower() == 'cifar100':
        model = Resnext29(num_classes=100)
    
    if config.dataset.lower() == 'tiny':
        model = Resnext29(num_classes=200)

    if len(config.resume_model_path) > 0:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        pretrained_dict = torch.load(config.resume_model_path,  map_location=lambda storage, loc: storage)
        for k, v in pretrained_dict.items():
            if k.startswith('module'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model


def Resnet50_init(config):
    if config.dataset == 'ImageNet':
        model = ResNet(Bottleneck, [3, 4, 6, 3],modified=False)
        if config.pretrain:
            settings = pretrained_settings['resnet50']['imagenet']
            resnet = load_pretrained(model, 1000, settings)
            return resnet
        else:
            return model
    if config.dataset == 'CUB':
        resnet = ResNet(Bottleneck, [3, 4, 6, 3],modified=True,modified_num_classes=200)
        if config.pretrain:            
            if len(config.resume_model_path) > 0:
                #resnet = torch.nn.DataParallel(model)
                #resnet.cuda()
                #resnet.load_state_dict(torch.load(config.resume_model_path))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                pretrained_dict = torch.load(config.resume_model_path,  map_location=lambda storage, loc: storage)
                for k, v in pretrained_dict.items():
                    name = k[7:]                # remove `module.`
                    new_state_dict[name] = v
                resnet.load_state_dict(new_state_dict)
            else:
                settings = pretrained_settings['resnet50']['imagenet']
                resnet = load_pretrained(resnet, 1000, settings)
            return resnet
        else:
            return resnet

def get_model_loss(config):
    model = None
    if config.model.lower() == 'resnext29':
        model = Resnext29_init(config)

    if config.model.lower() == 'resnet50':
        model = Resnet50_init(config)

    loss_fun = get_lossfunction(config)
    return model, loss_fun

def get_lossfunction(config):
    loss_fun = None
    if config.loss.lower() == 'crossentropy':
        loss_fun = F.cross_entropy
    return loss_fun
    
def get_opti_scheduler(config, model,train_loader=None):
    optimizer = None
    lr_scheduler = None
    if config.classifier_optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.classifier_learning_rate)
    if config.classifier_optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.classifier_learning_rate, weight_decay=5e-4, momentum=0.9, nesterov=True)
    
    if 'cifar10' in config.dataset.lower():
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
            step,
            config.epoch * len(train_loader),
            1,  # lr_lambda computes multiplicative factor
            1e-6 / config.classifier_learning_rate))
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epoch , eta_min=0)
    if 'cub' in config.dataset.lower():
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(config.lr_decay_step), gamma = config.lr_decay_gamma)
    return optimizer, lr_scheduler

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
