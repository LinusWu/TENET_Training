import sys
sys.path.append('./Data/')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import os
from _datasets import Cub2011, TinyImageNet
from Augmix import AugMix
import numpy as np


def to_tensor():
    def _to_tensor(image):
        if len(image.shape) == 3:
            return torch.from_numpy(
                image.transpose(2, 0, 1).astype(np.float32))
        else:
            return torch.from_numpy(image[None, :, :].astype(np.float32))

    return _to_tensor


def normalize(mean, std):
    mean = np.array(mean)
    std = np.array(std)

    def _normalize(image):
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - mean) / std
        return image

    return _normalize


def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

def ImageNet_data_init(config, Augmentation=True):
    preprocess_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    if config.is_ten_crop:
        val_loader = DataLoader(
            datasets.ImageFolder(config.test_data_path,
            preprocess_test),
            batch_size=1, shuffle=False,
            num_workers=8, pin_memory=True)
    else:
        val_loader = DataLoader(
            datasets.ImageFolder(config.test_data_path,
            preprocess_test),
            batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=True)

    if Augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])


        loader_origin = datasets.ImageFolder(config.raw_data_path, train_transform)

        loader1 = datasets.ImageFolder(config.aug_data_path['CAE'], train_aug_transform)
        loader2 = datasets.ImageFolder(config.aug_data_path['EDSR'], train_aug_transform)

        loader_list = torch.utils.data.ConcatDataset([loader1, loader2])

        loader_aug = AugMix(loader_list, config.aug_mix_severity)

        loader = torch.utils.data.ConcatDataset([loader_aug, loader_origin])

        train_loader = DataLoader(loader,batch_size=config.train_batch_size, shuffle=True, num_workers=8, pin_memory=True,drop_last=True)
    else:
        train_loader = DataLoader(
            datasets.ImageFolder(config.raw_data_path, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])),
            batch_size=config.train_batch_size, shuffle=True,
            num_workers=8, pin_memory=True,drop_last=True,)
    return train_loader, val_loader



def TinyImageNet_data_init(config, Augmentation=True):

    mean = [0.5] * 3
    std =  [0.5] * 3

    if Augmentation:
        train_aug_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        train_aug_transform_preprocess=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        loader_origin = datasets.ImageFolder(config.raw_data_path, train_aug_transform)
        class_to_idx = loader_origin.class_to_idx
        # loader1 = datasets.ImageFolder(config.aug_data_path['CAE'], train_aug_transform)
        # loader2 = datasets.ImageFolder(config.aug_data_path['EDSR'], train_aug_transform)

        # loader_list = torch.utils.data.ConcatDataset([loader1, loader2])
        loader_aug = AugMix(dataset=loader_origin,preprocess=train_aug_transform_preprocess ,aug_severity=config.aug_mix_severity,IMAGE_SIZE=64)

        #loader = torch.utils.data.ConcatDataset([loader_aug, loader_origin])

        train_loader = DataLoader(loader_aug,batch_size=config.train_batch_size, shuffle=True, num_workers=8, pin_memory=False,drop_last=True)
    else:
        train_dataset = datasets.ImageFolder(config.raw_data_path, transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]))
        class_to_idx = train_dataset.class_to_idx
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train_batch_size, shuffle=True,
            num_workers=8, pin_memory=False,drop_last=True,)
        
    if config.is_ten_crop:
        val_loader = DataLoader(
            TinyImageNet(root=config.test_data_path,path='val_annotations.txt',class_to_idx=class_to_idx,transform=preprocess_test),
            batch_size=1, shuffle=False,
            num_workers=8, pin_memory=False)
    else:
        val_loader = DataLoader(
            TinyImageNet(root=config.test_data_path,path='val_annotations.txt',class_to_idx=class_to_idx,transform=preprocess_test),
            batch_size=config.train_batch_size, shuffle=False,
            num_workers=8, pin_memory=False)

    return train_loader, val_loader

def CUB_data_init(config, Augmentation=True):
    
    train_aug_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomCrop(448),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

    train_aug_transform_preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
        ])
    if config.is_ten_crop:
        transform_test = transforms.Compose([
        transforms.Resize(512),
        transforms.TenCrop(448), # this is a list of PIL Images
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )(crop) for crop in crops])),
        ])
    else:
        transform_test = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((448,448)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
        ])

    train_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(448),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

    test_data = Cub2011(root=config.data_root, train=False, download=False,transform=transform_test)
    test_loader =  DataLoader(test_data, batch_size=config.test_batch_size,shuffle=True,drop_last=False, pin_memory=True,num_workers=8)
    
    if Augmentation:
        train_clean_data = Cub2011(root=config.data_root, train=True, download=False,transform=train_transform)

        train_aug_data = Cub2011(root=config.data_root, train=True, download=False,transform=train_aug_transform)
        #train_aug_data = AugMix(train_aug_data, config.aug_mix_severity, size=448)
        train_aug_data = AugMix(dataset=train_aug_data,preprocess=train_aug_transform_preprocess ,aug_severity=config.aug_mix_severity, IMAGE_SIZE=448)

        train_data = torch.utils.data.ConcatDataset([train_aug_data, train_clean_data])
        train_loader = DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True,drop_last=True, pin_memory=True,num_workers=8)
    else:
        train_data = Cub2011(root=config.data_root, train=True, download=False,transform=train_transform)
        train_loader =  DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True,drop_last=True, pin_memory=True,num_workers=8)
    return train_loader, test_loader



def CIFAR_data_init(config, name='cifar10',Augmentation=True, CutOut=False):
    mean = {
    'cifar10': (0.5, 0.5, 0.5),
    'cifar100': (0.5, 0.5, 0.5),
    }

    std = {
    'cifar10': (0.5, 0.5, 0.5),
    'cifar100': (0.5, 0.5, 0.5),
    }
    train_aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ])
    train_aug_transform_preprocess=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
        ])
    if config.is_ten_crop:
        transform_test = transforms.Compose([
        transforms.Resize(48),
        transforms.TenCrop(32), # this is a list of PIL Images
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean[name], std[name])(crop) for crop in crops])),
        ])
    else:
        transform_test = transforms.Compose([
            #transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean[name], std[name]),
        ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name]),
    ])
    if name.lower() == 'cifar10':
        train_clean_data = torchvision.datasets.CIFAR10(
            root=config.data_root+'/CIFAR10', train=True, download=False, transform=transform_train)
        test_data = torchvision.datasets.CIFAR10(
            root=config.data_root+'/CIFAR10', train=False, download=False, transform=transform_test)


    elif name.lower() == 'cifar100':
        train_clean_data = torchvision.datasets.CIFAR100(
            root=config.data_root+'/CIFAR100', train=True, download=False, transform=transform_train)
        test_data = torchvision.datasets.CIFAR100(
            root=config.data_root+'/CIFAR100', train=False, download=False, transform=transform_test)

    #test_data = Cub2011(root=config.data_root, train=False, download=False,transform=transform_test)
    test_loader =  DataLoader(test_data, batch_size=config.test_batch_size,shuffle=True, pin_memory=True,num_workers=8)
    
    if Augmentation:
        #train_clean_data = Cub2011(root=config.data_root, train=True, download=False,transform=train_transform)
        if name.lower() == 'cifar10':
            train_aug_data = torchvision.datasets.CIFAR10(root=config.data_root+'/CIFAR10', train=True, download=False, transform=train_aug_transform)
        elif name.lower() == 'cifar100':
            train_aug_data = torchvision.datasets.CIFAR100(root=config.data_root+'/CIFAR100', train=True, download=False, transform=train_aug_transform)
        #Cub2011(root=config.data_root, train=True, download=False,transform=train_aug_transform)
        # train_aug_data_old = AugMix(train_aug_data, config.aug_mix_severity,32, mean[name], std[name])
        if CutOut:
            print("CutOut")
            cutout_size = 16
            cutout_prob = 1
            cutout_inside = False
            transform_train_cutout = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(mean[name], std[name]),
                normalize(mean[name], std[name]),
                cutout(cutout_size, cutout_prob, cutout_inside),
                to_tensor()

            ])
            if name.lower() == 'cifar10':
                train_aug_data = torchvision.datasets.CIFAR10(root=config.data_root+'/CIFAR10', train=True, download=False, transform=transform_train_cutout)
            elif name.lower() == 'cifar100':
                train_aug_data = torchvision.datasets.CIFAR100(root=config.data_root+'/CIFAR100', train=True, download=False, transform=transform_train_cutout)
        else:
            train_aug_data = AugMix(dataset=train_aug_data,preprocess=train_aug_transform_preprocess ,aug_severity=config.aug_mix_severity)
        train_data = torch.utils.data.ConcatDataset([train_aug_data, train_clean_data])
        train_loader = DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True, pin_memory=True,num_workers=8)
        #train_loader = DataLoader(train_aug_data, batch_size=config.train_batch_size,shuffle=True,drop_last=True, pin_memory=False,num_workers=8)
    else:
        # train_data = train_clean_data
        if name.lower() == 'cifar10':
            train_data_copy = torchvision.datasets.CIFAR10(root=config.data_root+'/CIFAR10', train=True, download=False, transform=transform_train)
        elif name.lower() == 'cifar100':
            train_data_copy = torchvision.datasets.CIFAR100(root=config.data_root+'/CIFAR100', train=True, download=False, transform=transform_train)
        train_data = torch.utils.data.ConcatDataset([train_clean_data, train_data_copy])
        train_loader =  DataLoader(train_data, batch_size=config.train_batch_size,shuffle=True,drop_last=True, pin_memory=True,num_workers=8)
    return train_loader, test_loader


def get_data(config):
    train_loader = None
    test_loader = None
    if config.dataset.lower() == 'cifar10':
        train_loader, test_loader = CIFAR_data_init(config, 'cifar10',config.augmentation_switch)
    if config.dataset.lower() == 'cifar100':
        train_loader, test_loader = CIFAR_data_init(config, 'cifar100',config.augmentation_switch)
    if config.dataset.lower() == 'cub':
        train_loader, test_loader = CUB_data_init(config, config.augmentation_switch)
    return train_loader, test_loader