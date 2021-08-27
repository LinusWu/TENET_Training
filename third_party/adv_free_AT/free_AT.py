import torch

def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)